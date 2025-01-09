# -*- coding: utf-8 -*-
from flask import request, jsonify
from load_forecast_platform.api import app
from load_forecast_platform.utils.database import DataBase
from load_forecast_platform.utils.config import Config
from load_forecast_platform.data_processor.data_processor import DataProcessor
from load_forecast_platform.api.schemas import (
    StationRegisterRequest, RealTimeDataUploadRequest, ForecastRequest,
    FeatureSearchRequest, HyperParamSearchRequest, HistoryMeteoRequest,
    ModelTrainRequest, CommonResponse
)

import sys
from loguru import logger
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import text
import requests




# 1. 电站注册接口
@app.route('/ustlf/station/register', methods=['POST'])
def register_station():
    """电站注册接口"""
    try:
        # 验证请求数据
        data = request.get_json()
        station_data = StationRegisterRequest(**data)

        # 处理历史数据
        processor = DataProcessor()

        # 处理历史负荷数据
        load_df = pd.DataFrame(station_data.his_load)
        processed_load = processor.process_load_data(load_df)
        processed_load['site_id']= station_data.site_id
        processed_load['upload_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        processed_load['load_time'] = processed_load.index


        # 检查数据时长是否满足3个月
        # time_span = processed_load['load_time'].max() - processed_load['load_time'].min()
        time_span = processed_load.index.max() - processed_load.index.min()
        if time_span.days < 90:
            return jsonify(CommonResponse(
                code="402",
                msg="历史负荷时长不足3个月"
            ).model_dump())

        # 处理历史气象数据
        station_info_df = pd.DataFrame([{
            'site_id': station_data.site_id,
            'site_name': station_data.site_name,
            'longitude':station_data.longitude,
            'latitude':station_data.latitude,
            'stype':station_data.stype,
            'rated_capacity':station_data.rated_capacity,
            'rated_power':station_data.rated_power,
            'rated_power_pv':station_data.rated_power_pv,
            'frequency_load':station_data.frequency_load,
            'frequency_meteo':station_data.frequency_meteo,
            'first_load_time':processed_load.index[0].to_pydatetime().strftime('%Y-%m-%d %H:%M:%S'),
            'upload_time':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }])
        meteo_df = pd.DataFrame(station_data.his_meteo)
        processed_meteo = processor.process_meteo_data(meteo_df)
        processed_meteo['site_id']= station_data.site_id
        processed_meteo['update_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        processed_meteo['meteo_id'] = 1     # 默认气象id为1
        meteo_df.set_index('meteo_times', inplace=True)
        meteo_df['meteo_times'] = meteo_df.index
        # meteo_df['winds']

        # 存储数据
        try:
            logger.info("开始初始化数据库...")
            config = Config()
            logger.info("配置加载成功")
            logger.info(f"数据库配置: {config.database}")

            db = DataBase(**config.database)
            db.insert(table='ustlf_station_info', df=station_info_df)
            db.insert(table='ustlf_station_history_load', df=processed_load)
            db.insert(table='ustlf_station_meteo_data', df=processed_meteo)

        except Exception as e:
            logger.error(f"程序执行失败: {str(e)}")
            sys.exit(1)

    except Exception as e:
        return jsonify(CommonResponse(
            code="401",
            msg=f"参数验证失败: {str(e)}"
        ).model_dump())
    return jsonify(CommonResponse(
            code="200",
            msg=f"上传成功"
        ).model_dump())


# 2. 实时数据上传接口
@app.route('/ustlf/station/real_time_data_upload', methods=['POST'])
def upload_real_time_data():
    """实时数据上传接口"""
    try:
        # 验证请求数据
        data = request.get_json()
        upload_data = RealTimeDataUploadRequest(**data)

        # 处理数据
        processor = DataProcessor()

        # 处理实时负荷数据
        load_df = pd.DataFrame(upload_data.real_his_load)
        processed_load = processor.process_load_data(load_df)

        # 处理实时气象数据
        meteo_df = pd.DataFrame(upload_data.real_his_meteo)
        processed_meteo = processor.process_meteo_data(meteo_df)

        # 存储数据
        db = DataBase(Config().database)
        try:
            with db.engine.begin() as conn:
                # 获取电站名称
                site_query = "SELECT site_name FROM ustlf_station_info WHERE site_id = :site_id"
                result = conn.execute(text(site_query), {'site_id': upload_data.site_id}).fetchone()
                if not result:
                    return jsonify(CommonResponse(
                        code="401",
                        msg="电站不存在"
                    ).model_dump())
                site_name = result[0]

                # 存储实时负荷数据
                for _, row in processed_load.iterrows():
                    insert_load_sql = """
                        INSERT INTO ustlf_station_history_load (
                            site_id, site_name, load_time, load_data, upload_time
                        ) VALUES (
                            :site_id, :site_name, :load_time, :load_data, :upload_time
                        ) ON DUPLICATE KEY UPDATE
                            load_data = VALUES(load_data),
                            upload_time = VALUES(upload_time)
                    """
                    load_data = {
                        'site_id': upload_data.site_id,
                        'site_name': site_name,
                        'load_time': row['timestamp'],
                        'load_data': row['load'],
                        'upload_time': datetime.now()
                    }
                    conn.execute(text(insert_load_sql), load_data)

                # 存储实时气象数据
                for _, row in processed_meteo.iterrows():
                    insert_meteo_sql = """
                        INSERT INTO ustlf_station_meteo_data (
                            site_id, meteo_id, meteo_times, update_time,
                            relative_humidity_2m, surface_pressure, precipitation,
                            wind_speed_10m, temperature_2m, shortwave_radiation
                        ) VALUES (
                            :site_id, :meteo_id, :meteo_times, :update_time,
                            :relative_humidity_2m, :surface_pressure, :precipitation,
                            :wind_speed_10m, :temperature_2m, :shortwave_radiation
                        ) ON DUPLICATE KEY UPDATE
                            relative_humidity_2m = VALUES(relative_humidity_2m),
                            surface_pressure = VALUES(surface_pressure),
                            precipitation = VALUES(precipitation),
                            wind_speed_10m = VALUES(wind_speed_10m),
                            temperature_2m = VALUES(temperature_2m),
                            shortwave_radiation = VALUES(shortwave_radiation),
                            update_time = VALUES(update_time)
                    """
                    meteo_data = {
                        'site_id': upload_data.site_id,
                        'meteo_id': 1,  # 默认气象源ID
                        'meteo_times': row['timestamp'],
                        'update_time': datetime.now(),
                        'relative_humidity_2m': row.get('relative_humidity_2m'),
                        'surface_pressure': row.get('surface_pressure'),
                        'precipitation': row.get('precipitation'),
                        'wind_speed_10m': row.get('wind_speed_10m'),
                        'temperature_2m': row.get('temperature_2m'),
                        'shortwave_radiation': row.get('shortwave_radiation')
                    }
                    conn.execute(text(insert_meteo_sql), meteo_data)

            return jsonify(CommonResponse(
                code="200",
                msg="实时数据上传成功"
            ).model_dump())

        except Exception as e:
            return jsonify(CommonResponse(
                code="500",
                msg=f"数据存储失败: {str(e)}"
            ).model_dump())
        finally:
            db.close()

    except Exception as e:
        return jsonify(CommonResponse(
            code="401",
            msg=f"参数验证失败: {str(e)}"
        ).model_dump())


# 3. 预测结果拉取接口
@app.route('/ustlf/station/time_ustl_forcast_res', methods=['POST'])
def get_forecast_result():
    """预测结果拉取接口"""
    try:
        # 验证请求数据
        data = request.get_json()
        forecast_req = ForecastRequest(**data)

        # 验证开始时间格式
        try:
            begin_time = datetime.strptime(forecast_req.begin_time, "%H:%M")
            # 确保时间是15分钟的整数倍
            if begin_time.minute % 15 != 0:
                return jsonify(CommonResponse(
                    code="401",
                    msg="开始时间必须是15分钟的整数倍"
                ).model_dump())
        except ValueError:
            return jsonify(CommonResponse(
                code="401",
                msg="时间格式错误，应为HH:MM格式"
            ).model_dump())

        # 查询预测结果
        db = DataBase(Config().database)
        try:
            with db.engine.begin() as conn:
                # 查询最新的预测结果
                query = """
                    SELECT res_data 
                    FROM ustlf_pred_res 
                    WHERE site_id = :site_id 
                    AND forcast_time_start = :begin_time
                    ORDER BY cal_time DESC 
                    LIMIT 1
                """
                result = conn.execute(text(query), {
                    'site_id': forecast_req.site_id,
                    'begin_time': forecast_req.begin_time
                }).fetchone()

                if not result:
                    return jsonify(CommonResponse(
                        code="401",
                        msg="未找到预测结果"
                    ).model_dump())

                return jsonify(CommonResponse(
                    code="200",
                    msg="获取预测结果成功",
                    data={"res": result[0]}
                ).model_dump())

        except Exception as e:
            return jsonify(CommonResponse(
                code="500",
                msg=f"查询失败: {str(e)}"
            ).model_dump())
        finally:
            db.close()

    except Exception as e:
        return jsonify(CommonResponse(
            code="401",
            msg=f"参数验证失败: {str(e)}"
        ).model_dump())


# 4. 输入特征搜索接口
@app.route('/ustlf/station/feature_search', methods=['POST'])
def feature_search():
    """输入特征搜索接口"""
    try:
        # 验证请求数据
        data = request.get_json()
        search_req = FeatureSearchRequest(**data)

        db = DataBase(Config().database)
        try:
            with db.engine.begin() as conn:
                # 1. 获取历史负荷数据
                load_query = """
                    SELECT load_time, load_data
                    FROM ustlf_station_history_load
                    WHERE site_id = :site_id
                    ORDER BY load_time DESC
                """
                load_df = pd.read_sql(load_query, conn, params={'site_id': search_req.site_id})

                # 2. 获取历史气象数据
                meteo_query = """
                    SELECT meteo_times, relative_humidity_2m, surface_pressure, 
                           precipitation, wind_speed_10m, temperature_2m, shortwave_radiation
                    FROM ustlf_station_meteo_data
                    WHERE site_id = :site_id
                    ORDER BY meteo_times DESC
                """
                meteo_df = pd.read_sql(meteo_query, conn, params={'site_id': search_req.site_id})

                # 3. 特征工程处理
                processor = DataProcessor()
                features = processor.extract_features(load_df, meteo_df)

                # 4. 存储特征信息
                feature_sql = """
                    INSERT INTO ustlf_model_feature_hp_info (
                        site_id, feature_info, hyperparams_info
                    ) VALUES (
                        :site_id, :feature_info, :hyperparams_info
                    ) ON DUPLICATE KEY UPDATE
                        feature_info = VALUES(feature_info)
                """
                conn.execute(text(feature_sql), {
                    'site_id': search_req.site_id,
                    'feature_info': str(features),
                    'hyperparams_info': '{}'  # 初始化空的超参数信息
                })

                return jsonify(CommonResponse(
                    code="200",
                    msg="特征搜索成功",
                    data={"features": features}
                ).model_dump())

        except Exception as e:
            return jsonify(CommonResponse(
                code="500",
                msg=f"特征搜索失败: {str(e)}"
            ).model_dump())
        finally:
            db.close()

    except Exception as e:
        return jsonify(CommonResponse(
            code="401",
            msg=f"参数验证失败: {str(e)}"
        ).model_dump())


# 5. 超参数搜索接口
@app.route('/ustlf/station/hp_search', methods=['POST'])
def hyperparameter_search():
    """超参数搜索接口"""
    try:
        data = request.get_json()
        hp_req = HyperParamSearchRequest(**data)

        db = DataBase(Config().database)
        try:
            with db.engine.begin() as conn:
                # 1. 获取特征信息
                feature_query = """
                    SELECT feature_info
                    FROM ustlf_model_feature_hp_info
                    WHERE site_id = :site_id
                """
                feature_result = conn.execute(text(feature_query),
                                              {'site_id': hp_req.site_id}).fetchone()

                if not feature_result:
                    return jsonify(CommonResponse(
                        code="401",
                        msg="未找到特征信息，请先进行特征搜索"
                    ).model_dump())

                # 2. 执行超参数搜索
                from load_forecast_platform.trainer.param_optimizer import ParamOptimizer
                optimizer = ParamOptimizer()
                best_params = optimizer.search(feature_result[0])

                # 3. 更新超参数信息
                update_sql = """
                    UPDATE ustlf_model_feature_hp_info
                    SET hyperparams_info = :hyperparams_info
                    WHERE site_id = :site_id
                """
                conn.execute(text(update_sql), {
                    'site_id': hp_req.site_id,
                    'hyperparams_info': str(best_params)
                })

                return jsonify(CommonResponse(
                    code="200",
                    msg="超参数搜索成功",
                    data={"hyperparameters": best_params}
                ).model_dump())

        except Exception as e:
            return jsonify(CommonResponse(
                code="500",
                msg=f"超参数搜索失败: {str(e)}"
            ).model_dump())
        finally:
            db.close()

    except Exception as e:
        return jsonify(CommonResponse(
            code="401",
            msg=f"参数验证失败: {str(e)}"
        ).model_dump())


# 6. 历史气象拉取接口
@app.route('/ustlf/station/get_history_meteo', methods=['POST'])
def get_history_meteo():
    """历史气象拉取接口"""
    try:
        data = request.get_json()
        meteo_req = HistoryMeteoRequest(**data)

        # # 获取数据库连接
        logger.info("开始初始化数据库...")
        config = Config()
        logger.info("配置加载成功")
        logger.info(f"数据库配置: {config.database}")
        db = DataBase(**config.database)

        sql_query = """
            SELECT latitude, longitude 
            FROM ustlf_station_info 
            WHERE site_id = '{}'
        """.format(meteo_req.site_id)

        site_info = db.query(sql_query)
        latitude = float(site_info['latitude'])
        longitude = float(site_info['longitude'])

        # 处理时间参数
        current_time = datetime.now()
        end_time = current_time.replace(hour=19, minute=0, second=0, microsecond=0)
        # 如果没有提供开始时间，使用默认值（当前日之前的3天）
        if not meteo_req.start_time:
            start_time = (end_time - timedelta(days=3)).strftime('%Y-%m-%d')
        else:
            start_time = meteo_req.start_time

        # 如果没有提供结束时间，使用默认值（当日19点）
        if not meteo_req.end_time:
            end_time = end_time.strftime('%Y-%m-%d')
        else:
            end_time = meteo_req.end_time

        start_time = str(start_time)
        end_time = str(end_time)

        # 调用气象API
        url = "http://8.212.49.208:5009/weather"
        headers = {'apikey': 'renewable_api_key'}
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'timezone': 'Asia/Shanghai',
            'forecast_days': 3,
            'interval': '15T',
            'start_date': start_time,
            'end_date': end_time,
            'par': 'shortwave_radiation,temperature_2m,relative_humidity_2m,surface_pressure,precipitation,wind_speed_10m'
        }

        response = requests.get(url=url, params=params, headers=headers)
        res = response.json()
        data = res['par']

        # 处理API返回数据
        temperature = data['temperature_2m']
        pressure = data['surface_pressure']
        humidity = data['relative_humidity_2m']
        irradiation = data['shortwave_radiation']
        precipitation = data['precipitation']
        wind_speed_10m = data['wind_speed_10m']
        timestamp = data['time']

        data_dict = {
            'meteo_times':timestamp,
            'temperature_2m': temperature,
            'surface_pressure': pressure,
            'relative_humidity_2m': humidity,
            'shortwave_radiation': irradiation,
            'wind_speed_10m': wind_speed_10m,
            'precipitation': precipitation
        }

        # 创建DataFrame并添加站点信息
        df = pd.DataFrame(data_dict)
        df['site_id'] = meteo_req.site_id
        df['meteo_id'] = meteo_req.meteo_id
        df['update_time'] = current_time.strftime('%Y-%m-%d %H:%M:%S')

        # 存储到数据库

        db.insert(table='ustlf_station_meteo_data',df=df)


    except Exception as e:
        return jsonify(CommonResponse(
            code="401",
            msg=f"参数验证失败: {str(e)}"
        ).model_dump())

    return jsonify(CommonResponse(
            code="200",
            msg=f"历史气象拉取成功"
        ).model_dump())

# 7. 模型训练接口
@app.route('/ustlf/station/model_train', methods=['POST'])
def train_model():
    """模型训练接口"""
    try:
        data = request.get_json()
        train_req = ModelTrainRequest(**data)

        db = DataBase(Config().database)
        try:
            with db.engine.begin() as conn:
                # 1. 获取特征和超参数信息
                info_query = """
                    SELECT feature_info, hyperparams_info
                    FROM ustlf_model_feature_hp_info
                    WHERE site_id = :site_id
                """
                info_result = conn.execute(text(info_query),
                                           {'site_id': train_req.site_id}).fetchone()

                if not info_result:
                    return jsonify(CommonResponse(
                        code="401",
                        msg="未找到特征和超参数信息，请先进行特征和超参数搜索"
                    ).model_dump())

                feature_info = eval(info_result[0])
                hyperparams = eval(info_result[1])

                # 2. 获取训练数据
                load_query = """
                    SELECT load_time, load_data
                    FROM ustlf_station_history_load
                    WHERE site_id = :site_id
                    ORDER BY load_time
                """
                load_df = pd.read_sql(load_query, conn, params={'site_id': train_req.site_id})

                meteo_query = """
                    SELECT *
                    FROM ustlf_station_meteo_data
                    WHERE site_id = :site_id
                    ORDER BY meteo_times
                """
                meteo_df = pd.read_sql(meteo_query, conn, params={'site_id': train_req.site_id})

                # 3. 训练模型
                from load_forecast_platform.trainer.model_trainer import ModelTrainer
                trainer = ModelTrainer(feature_info, hyperparams)
                model = trainer.train(load_df, meteo_df)

                # 4. 保存模型
                model_path = f"models/site_{train_req.site_id}_model.pkl"
                model.save(model_path)

                return jsonify(CommonResponse(
                    code="200",
                    msg="模型训练成功",
                    data={"model_path": model_path}
                ).model_dump())

        except Exception as e:
            return jsonify(CommonResponse(
                code="500",
                msg=f"模型训练失败: {str(e)}"
            ).model_dump())
        finally:
            db.close()

    except Exception as e:
        return jsonify(CommonResponse(
            code="401",
            msg=f"参数验证失败: {str(e)}"
        ).model_dump())

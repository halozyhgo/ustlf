# -*- coding: utf-8 -*-
import json
import os

from flask import request, jsonify
from loadFcst.utils import feature_engine
from sympy.physics.units import years

from load_forecast_platform.models.lightgbm_model import ML_Model
from load_forecast_platform.api import app
from load_forecast_platform.data_processor.feature_engineer import FeatureEngineer
from load_forecast_platform.utils.database import DataBase
from load_forecast_platform.utils.config import Config
from load_forecast_platform.data_processor.data_processor import DataProcessor
from load_forecast_platform.api.schemas import (
    StationRegisterRequest, RealTimeDataUploadRequest, ForecastRequest,
    FeatureSearchRequest, HyperParamSearchRequest, HistoryMeteoRequest,
    ModelTrainRequest, CommonResponse, ForecastMeteoRequest
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
        load_df['load_data'] = pd.to_numeric(load_df['load_data'],errors='coerce')
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
    '''
            todo:
            数据准备
            0、获取最新的历史负荷数据
            1、从数据库中拉取过去8天的负荷数据
            2、从数据库中拉取过去8天的气象数据
            3、从气象源地址拉取未来3天的气象数据

            根据预测尺度获取不同预测尺度的输入数据

            4、将负荷数据与历史气象合并，获取对应，然后获取对应的输入特征
            5、将对应时间的预测气象换为从气象源拉取的气象

            加载预测模型
            1、从models_pkl文件中获取对应电站的模型，然后载对应电站id且预测第几个点的模型


            2、每个模型进行预测并输出
            3、按照时间维度拼接在一起
            4、汇总成16个点的预测结果
            5、存回数据库 [site_id,begin_time,res_data]
            6、返回结果
            '''

    try:
        # 验证请求数据
        data = request.get_json()
        upload_data = RealTimeDataUploadRequest(**data)

        # 处理数据
        new_data = upload_data.real_his_load
        new_data = pd.DataFrame(new_data)


        new_data = pd.DataFrame(upload_data.real_his_load)
        new_data.set_index(pd.to_datetime(new_data['load_time']), inplace=True)
        new_data['load_data'] = pd.to_numeric(new_data['load_data'], errors='coerce')
        new_data['site_id'] = upload_data.site_id
        new_data['upload_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_data['load_time'] = new_data.index

        config = Config()
        model_param = config.config['model_params']['lightgbm']
        # train_param = config['lightgbm_training']
        db = DataBase(**config.database)

        # todo: 将新数据存入到数据库中
        db.insert('ustlf_station_history_load', new_data)

        # 1、从表中获取模型超参数和输入特征
        info_query = """SELECT feature_info, hyperparams_info
                        FROM ustlf_model_feature_hp_info
                        WHERE site_id = '{}'
                        """.format(upload_data.site_id)
        res = db.query(info_query)
        if len(res)==0:
            feature_info_table = None
            hyperparams_info_table = None
        else:
            feature_info_table = res['feature_info']
            hyperparams_info_table = res['hyperparams_info']
        hyperparams = model_param
        if hyperparams_info_table:
            hyperparams = hyperparams_info_table

        load_end_time = datetime.now()

        # 先以2024年9月2日做测试
        # load_end_time = datetime.now().replace(year=2024,month=9,day=2,hour=7,minute=45)
        load_end_time = new_data.index[-1]
        load_start_time = load_end_time-timedelta(days=9)

        load_start_time = load_start_time.strftime('%Y-%m-%d %H:%M:%S')
        load_end_time = load_end_time.strftime('%Y-%m-%d %H:%M:%S')

        # 2、从表中获取训练负荷数据
        # todo:获取历史负荷过去9天的负荷数据
        load_query = """
                            SELECT load_time, load_data
                            FROM ustlf_station_history_load
                            WHERE site_id = '{}' AND load_time >='{}' AND load_time<='{}'
                            ORDER BY load_time
                        """.format(upload_data.site_id,load_start_time, load_end_time)
        load_df = db.query(load_query)
        # load_df.set_index(pd.to_datetime(load_df['load_time']), inplace=True)

        # 3、从预测气象表中获取以后4H预测气象
        # todo: 获取传输数据最新时间以后的4H数据，如果传输为为8点的数据，则返回8点15 ~ 12点15的预测气象数据
        meteo_start_time = new_data.index[-1]
        meteo_start_time = meteo_start_time+timedelta(minutes=15)
        meteo_end_time = meteo_start_time + timedelta(hours=4)
        meteo_start_time = meteo_start_time.strftime('%Y-%m-%d %H:%M:%S')
        meteo_end_time = meteo_end_time.strftime('%Y-%m-%d %H:%M:%S')
        #
        # todo:获取预测气象，当有meteo_times相同的情况下只使用upload_time最新的那一条,通过去重保留第一条完成
        meteo_query = """
                    SELECT *
                    FROM ustlf_station_forecast_meteo_data
                    WHERE site_id = '{}' AND meteo_times >= '{}' AND meteo_times < '{}'
                    ORDER BY meteo_times, update_time DESC
                    """.format(upload_data.site_id, meteo_start_time, meteo_end_time)
        meteo_df = db.query(meteo_query)
        meteo_df.set_index(pd.to_datetime(meteo_df['meteo_times']), inplace=True)
        meteo_df = meteo_df[~meteo_df.index.duplicated(keep='first')]

        # todo: 将气象数据与负荷数据结合，然后根据现在的特征组合，分类出输入特征和标签
        H_list = [i * 0.25 for i in range(1, 17)]
        # todo: 获取预测时间的时间戳
        pred_time = new_data.index[-1]
        forcast_time_start = pred_time + timedelta(minutes=15)
        mydict = {}
        for i,H in enumerate(H_list):
            # 自增15min
            pred_time = pred_time + timedelta(minutes=15)
            # todo : 1、获取对应模型
            model_path = "./models_pkl/site_id_{}".format(upload_data.site_id)
            lgb_model = ML_Model(model_name='site_id_{}_model_{}'.format(upload_data.site_id, i),
                                 model_params=hyperparams, model_path=model_path)
            lgb_model.load_model()

            # todo : 2、获取模型输入数据
            feature_engine = FeatureEngineer()
            df_i, feature_columns = feature_engine.extract_pred_features(load_df, meteo_df, H)

            input_features = feature_columns
            if feature_info_table:
                input_features = feature_info_table
            input_feature = list(set(input_features))
            input_feature.sort()
            input_data = df_i.loc[pred_time.strftime('%Y-%m-%d %H:%M:%S')][input_feature]
            pred_i = lgb_model.model_predict([input_data])
            mydict[pred_time.strftime('%Y-%m-%d %H:%M:%S')] = round(float(pred_i),2)

        # 4、将预测结果存入数据库中
        res = {'site_id': upload_data.site_id, 'meteo_id': 1, 'cal_time': datetime.now(),
               'forcast_time_start': forcast_time_start, 'res_data': [json.dumps(mydict)]}
        pred_df = pd.DataFrame(res)
        db.insert(table='ustlf_pred_res', df=pred_df)
        return jsonify(CommonResponse(
            code="200",
            msg="实时数据上传成功,预测结果返回",
            data=mydict
        ).model_dump())

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

        # # 获取数据库连接
        logger.info("开始初始化数据库...")
        config = Config()
        logger.info("配置加载成功")
        logger.info(f"数据库配置: {config.database}")
        db = DataBase(**config.database)

        query_sql =  '''
        SELECT res_data FROM `vpp`.`ustlf_pred_res` 
        WHERE `site_id` LIKE '%{}%' AND `forcast_time_start` LIKE '%{}%' 
        ORDER BY `cal_time` DESC LIMIT 1;
        '''.format(forecast_req.site_id, forecast_req.forcast_time_start)

        red_df = db.query(query_sql)
        if red_df.empty:
            return jsonify(CommonResponse(
                code="401",
                msg=f"参数验证失败"
            ).model_dump())
        else:
            return jsonify(CommonResponse(
                code="200",
                msg="预测结果拉取成功",
                data=json.loads(red_df.iloc[0]['res_data'])
            ).model_dump())

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
    '''
    0、建立数据库连接（参考其他接口中的连接方式）
    1、从数据库中获取历史负荷数据
    2、从数据库中获取历史气象数据（1~2可以参考模型训练接口代码 train_model方法）
    3、获取基础的输入特征（范围比较大）
    4、获取基础的超参数组合区间（根据yaml的基础配置来生成超参数搜索的空间）
    5、对数据集分割，一部分做训练一部分用于对特征和超参数组合做验证
    6、得到最好的参数组合与特征组合，并将其保存到表ustlf_model_feature_hp_info中
    7、将模型保存到起来（保存的方式也可参考模型训练接口代码 train_model方法）
    
    '''



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

        config = Config()
        model_param = config.config['model_params']['lightgbm']
        # train_param = config['lightgbm_training']

        db = DataBase(**config.database)
        # 1、从表中获取模型超参数和输入特征
        info_query = """SELECT feature_info, hyperparams_info
                        FROM ustlf_model_feature_hp_info
                        WHERE site_id = '{}'
                        """.format(train_req.site_id)
        res = db.query(info_query)
        if len(res)==0:
            feature_info_table = None
            hyperparams_info_table = None
        else:
            feature_info_table = res['feature_info']
            hyperparams_info_table = res['hyperparams_info']
        hyperparams = model_param
        if hyperparams_info_table:
            hyperparams = hyperparams_info_table

        end_date = datetime.now().strftime('%Y-%m-%d')
        if train_req.end_date:
            end_date = train_req.end_date


        # 2、从表中获取训练数据（）
        # todo:获取历史负荷
        load_query = """
                            SELECT load_time, load_data
                            FROM ustlf_station_history_load
                            WHERE site_id = '{}' AND load_time<'{}'
                            ORDER BY load_time
                        """.format(train_req.site_id,end_date)
        load_df = db.query(load_query)

        # todo:获取气象负荷
        meteo_query = """
                            SELECT *
                            FROM ustlf_station_meteo_data
                            WHERE site_id = '{}' AND meteo_times <'{}'
                            ORDER BY meteo_times
                        """.format(train_req.site_id,end_date)
        meteo_df = db.query(meteo_query)

        # todo: 将气象数据与负荷数据结合，然后根据现在的特征组合，分类出输入特征和标签，
        H_list = [i * 0.25 for i in range(1, 17)]
        df_list = []
        for H in H_list:
            feature_engine = FeatureEngineer()
            df_i,feature_columns = feature_engine.extract_features(load_df, meteo_df,h=H)
            df_i_copy = df_i.copy()
            df_list.append(df_i_copy)
        input_feature = feature_columns
        if feature_info_table:  # 如果取得模型输入参数，更改为从数据库中获取的模型
            input_feature = feature_info_table

        input_feature = list(set(input_feature))
        input_feature.sort()
        # todo: 对list中的每个数据集进行训练，保存模型以电站id为目录名，i表示预测的未来第几个负荷值的模型
        for i,df in enumerate(df_list):
            df = df[:-96]
            X_train,Y_train = df[input_feature], df['load_data']
            model_path = "./models_pkl/site_id_{}".format(train_req.site_id)
            lgb_model = ML_Model(model_name='site_id_{}_model_{}'.format(train_req.site_id,i),model_params=hyperparams,model_path=model_path)
            lgb_model.get_model()
            lgb_model.model_train(X_train,Y_train)
            lgb_model.save_model()

        return jsonify(CommonResponse(
            code="200",
            msg="模型训练成功",
            data={"model_path保存到文件中": model_path}
        ).model_dump())

    except Exception as e:
        return jsonify(CommonResponse(
            code="500",
            msg=f"模型训练失败: {str(e)}"
        ).model_dump())



# 8. 预测气象保存
@app.route('/ustlf/station/get_forcast_meteo', methods=['POST'])
def get_forcast_meteo():
    """预测气象拉取接口"""
    try:
        data = request.get_json()
        meteo_req = ForecastMeteoRequest(**data)

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

        # 调用气象API
        url = "http://8.212.49.208:5009/weather"
        headers = {'apikey': 'renewable_api_key'}
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'timezone': 'Asia/Shanghai',
            'forecast_days': 4,
            'interval': '15T',
            'start_date': '2024-09-01',
            'end_date': '2024-09-06',
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

        db.insert(table='ustlf_station_forecast_meteo_data',df=df)


    except Exception as e:
        return jsonify(CommonResponse(
            code="401",
            msg=f"参数验证失败: {str(e)}"
        ).model_dump())

    return jsonify(CommonResponse(
            code="200",
            msg=f"预测气象拉取成功"
        ).model_dump())
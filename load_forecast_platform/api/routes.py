# -*- coding: utf-8 -*-
import json
import os
import itertools
import time
import sys
sys.path.append("..")

import tqdm
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error

from flask import Blueprint, request, jsonify

from models.lightgbm_model import ML_Model

from multiprocessing import Pool

# 创建蓝本
api_bp = Blueprint('api', __name__)
from data_processor.feature_engineer import FeatureEngineer
from utils.database import DataBase
from utils.config import Config
from data_processor.data_processor import DataProcessor
from api.schemas import (
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
import asyncio



# 1. 电站注册接口
@api_bp.route('/station/register', methods=['POST'])
def register_station():
    """电站注册接口"""
    logger.info("开始电站注册...")
    try:
        # 验证请求数据
        data = request.get_json()
        station_data = StationRegisterRequest(**data)

        # 处理历史负荷数据
        processor = DataProcessor()

        # 处理历史负荷数据
        load_df = pd.DataFrame(station_data.his_load)
        load_df['load_data'] = pd.to_numeric(load_df['load_data'],errors='coerce')
        # 根据额定发电功率与储能计划功率确定负荷下下限
        # 根据变压器额定容量确定负荷上限
        processed_load = processor.process_load_data(load_df,station_data.rated_transform_power,station_data.rated_power,station_data.rated_power_pv)
        processed_load['site_id']= station_data.site_id
        processed_load['upload_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        processed_load['load_time'] = processed_load.index


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
            'rated_transform_power':station_data.rated_transform_power,
            'first_load_time':processed_load.index[0].to_pydatetime().strftime('%Y-%m-%d %H:%M:%S'),
            'upload_time':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }])
        # meteo_df = pd.DataFrame(station_data.his_meteo)
        # processed_meteo = processor.process_meteo_data(meteo_df)
        # processed_meteo['site_id']= station_data.site_id
        # processed_meteo['update_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # processed_meteo['meteo_id'] = 1     # 默认气象id为1
        # meteo_df.set_index('meteo_times', inplace=True)
        # meteo_df['meteo_times'] = meteo_df.index


        # 存储数据
        try:
            logger.info("开始初始化数据库...")
            config = Config()
            logger.info("配置加载成功")
            logger.info(f"数据库配置: {config.database}")

            db = DataBase(**config.database)
            db.insert(table='ustlf_station_info', df=station_info_df)
            db.insert(table='ustlf_station_history_load', df=processed_load)
            meteo_request_dict={
                'site_id': station_data.site_id,
                'start_time':processed_load.index[0].to_pydatetime().strftime('%Y-%m-%d'),
                'end_time':processed_load.index[-1].to_pydatetime().strftime('%Y-%m-%d'),
            }
            get_history_meteo_method(meteo_request_dict)

            # 检查数据时长是否满足3个月
            time_span = processed_load.index.max() - processed_load.index.min()
            if time_span.days < 90:
                logger.warning(f"电站{station_data.site_id}历史负荷时长不足3个月")
            else:
                # 模型训练
                logger.info(f"电站{station_data.site_id}历史负荷时长超过3个月，开始模型训练")
                train_model_method({'site_id':station_data.site_id})


        except Exception as e:
            logger.error(f"程序执行失败: {str(e)}")
            sys.exit(1)
        logger.info(f"电站 {station_data.site_id} 注册成功")
    except Exception as e:
        logger.error(f"电站注册失败: {str(e)}")
        return jsonify(CommonResponse(
            code="401",
            msg=f"参数验证失败: {str(e)}"
        ).model_dump())
    return jsonify(CommonResponse(
            code="200",
            msg=f"上传成功"
        ).model_dump())

def real_time_pred(data):
    upload_data = RealTimeDataUploadRequest(**data)
    # time.sleep(3)
    config = Config()
    db = DataBase(**config.database)
    query = f"SELECT * FROM ustlf_station_info WHERE site_id = '{upload_data.site_id}'"
    station_info = db.query(query)

    if station_info.empty:
        logger.error(f"电站 {upload_data.site_id} 不存在")
        return {
            'code': '404',
            'msg': f'电站{upload_data.site_id}不存在'
        }
        # jsonify(CommonResponse(code="404", msg="电站不存在").model_dump())
    trained = station_info['trained'].iloc[0]

    # 处理数据
    # new_data = pd.DataFrame(upload_data.real_his_load)
    data = upload_data.real_his_load

    # 提取 load_time 和 load_data
    load_times = [item.load_time for item in data]
    load_data = [item.load_data for item in data]

    # 创建 DataFrame
    new_data = pd.DataFrame({
        'load_time': load_times,
        'load_data': load_data
    })

    new_data.set_index(pd.to_datetime(new_data['load_time']), inplace=True)
    new_data['load_data'] = pd.to_numeric(new_data['load_data'], errors='coerce')
    new_data['site_id'] = upload_data.site_id
    new_data['upload_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_data['load_time'] = new_data.index

    # todo:判断数据中是否含有异常数据，
    #  如果含有正常范围值之外的数据，报异常，并返回数据异常信号
    #  1、查询表ustlf_station_info中的rated_power 与 rated_power_pv，作为负荷下界；查询rated_transform_power，作为负荷上限
    #  2、判断数据是否在合理范围，不在合理范围则返回异常信号
    rated_power = station_info['rated_power'].iloc[0]  # 储能额定容量
    rated_power_pv = station_info['rated_power_pv'].iloc[0]  # 光伏发电功率
    rated_transform_power = station_info['rated_transform_power'].iloc[0]

    if (new_data['load_data'] < (rated_power_pv - rated_power)).any() or (
            new_data['load_data'] > rated_transform_power).any():
        logger.warning(f"电站 {upload_data.site_id} 的历史负荷数据异常")
        return {
            'code': '400',
            'msg': f'电站{upload_data.site_id}的历史负荷数据异常'
        }

    # todo: start 若数据正常则将新数据存入到数据库中

    db.insert('ustlf_station_history_load', new_data)
    # todo: end 将新数据存入到数据库中完成

    # todo: 获取历史负荷数据长度 判断历史负荷长度是否负荷要求
    load_query = f"""
                        SELECT load_time FROM ustlf_station_history_load
                        WHERE site_id = '{upload_data.site_id}'
                    """
    load_data = db.query(load_query)

    # todo：如果长度不符合预测模型要求，则返回错误
    if len(load_data) < 96 * 90:
        logger.warning(f"电站 {upload_data.site_id} 的历史负荷数据为空")
        return jsonify(CommonResponse(code="402", msg="历史负荷长度不足").model_dump())

    time_span = load_data['load_time'].max() - load_data['load_time'].min()

    if trained == 1:
        logger.info(f"电站 {upload_data.site_id} 已训练，开始预测...")
        # 调用模型进行预测
        with Pool(processes=10) as pool:
            # prediction_result = pool.map(predict_future_load,([(upload_data.site_id, new_data)]))
            prediction_result = predict_future_load(upload_data.site_id, new_data)
            return jsonify(
                CommonResponse(code="200", msg="数据已收到，预测结果返回", data=prediction_result).model_dump())
    else:
        if time_span.days < 90:
            logger.warning(f"电站 {upload_data.site_id} 的历史负荷长度不足3个月")
            return jsonify(CommonResponse(code="402", msg="历史负荷长度不足").model_dump())
        else:
            logger.info(f"电站 {upload_data.site_id} 历史负荷长度满足3个月，开始训练...")
            # 进行模型训练
            train_model_method({'site_id': upload_data.site_id})
            # 训练完成后，更新 trained 标志位
            update_query = f"UPDATE ustlf_station_info SET trained = 1 WHERE site_id = '{upload_data.site_id}'"
            db.execute(update_query)
            logger.info(f"电站 {upload_data.site_id} 模型训练完成，标记为已训练")
            # 进行预测
            prediction_result = predict_future_load(upload_data.site_id, new_data)
            return jsonify(
                CommonResponse(code="200", msg="数据已收到，预测结果返回", data=prediction_result).model_dump())


# 2. 实时数据上传接口
@api_bp.route('/station/real_time_data_upload', methods=['POST'])
def upload_real_time_data():
    logger.info("开始实时数据上传...")
    try:

        # todo：连接数据库
        config = Config()
        db = DataBase(**config.database)
        # 验证请求数据
        data = request.get_json()
        res = real_time_pred(data)
        print(res)
        return res


    except Exception as e:
        logger.error(f"实时数据上传失败: {str(e)}")
        return jsonify(CommonResponse(code="500", msg=f"实时数据上传失败: {str(e)}").model_dump())

def predict_future_load(site_id, new_data):
    # site_id = req[0]
    # new_data = req[1]
    logger.info(f"开始对电站 {site_id} 进行负荷预测...")
    config = Config()
    model_param = config.config['model_params']['lightgbm']
    db = DataBase(**config.database)

    # 1、从表中获取模型超参数和输入特征
    info_query = """SELECT feature_info, hyperparams_info
                    FROM ustlf_model_feature_hp_info
                    WHERE site_id = '{}'
                    """.format(site_id)
    res = db.query(info_query)
    if len(res)==0:
        feature_info_table = None
        hyperparams_info_table = None
        logger.warning(f"电站 {site_id} 的模型信息不存在")
    else:
        feature_info_table = res['feature_info']
        hyperparams_info_table = res['hyperparams_info']
    hyperparams = model_param
    if hyperparams_info_table:
        hyperparams = hyperparams_info_table

    # todo： 实际部署后使用该方式
    # load_end_time = datetime.now()
    load_end_time = new_data.index[-1]

    # todo：先以2024年9月2日做测试
    # load_end_time = datetime.now().replace(year=2024,month=9,day=2,hour=7,minute=45)
    # load_end_time = new_data.index[-1]
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
                    """.format(site_id,load_start_time, load_end_time)
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
                """.format(site_id, meteo_start_time, meteo_end_time)
    meteo_df = db.query(meteo_query)
    if len(meteo_df)<16:
        logger.warning(f"电站{site_id}预测气象数据不足,现在重新拉取气象预测数据")
        # 进行调用拉取预测气象的接口
        get_forcast_meteo_method({"site_id":site_id,'start_time':meteo_start_time[:10],'end_time':meteo_end_time[:10]})
        meteo_df = db.query(meteo_query)
    if len(meteo_df)<16:
        logger.warning(f"电站{site_id}预测气象数据不足，拉取后仍然不足")
        # return jsonify(CommonResponse(code="402", msg="预测气象数据不足").model_dump())

    meteo_df.set_index(pd.to_datetime(meteo_df['meteo_times']), inplace=True)
    meteo_df = meteo_df[~meteo_df.index.duplicated(keep='first')]
    if len(meteo_df)<16:
        logger.warning(f"电站{site_id}预测气象数据不足,现在重新拉取气象预测数据")
        # 进行调用拉取预测气象的接口
        get_forcast_meteo_method({"site_id":site_id,'start_time':meteo_start_time[:10],'end_time':meteo_end_time[:10]})
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
        model_path = "./models_pkl/site_id_{}".format(site_id)
        lgb_model = ML_Model(model_name='site_id_{}_model_{}'.format(site_id, i),
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
    res = {'site_id': site_id, 'meteo_id': 1, 'cal_time': datetime.now(),
           'forcast_time_start': forcast_time_start, 'res_data': [json.dumps(mydict)]}
    pred_df = pd.DataFrame(res)
    db.insert(table='ustlf_pred_res', df=pred_df)
    return mydict

        # return jsonify(CommonResponse(
        #     code="200",
        #     msg="实时数据上传成功,预测结果返回",
        #     data=mydict
        # ).model_dump())

# @api_bp.route('/station/hello_world', methods=['GET'])
# def get_forecast_result():
#     print('hello_world')
#     jsonify(CommonResponse(
#             code="401",
#             msg=f"参数验证失败: {str(e)}"
#         ).model_dump())

# 3. 预测结果拉取接口
@api_bp.route('/station/time_ustl_forcast_res', methods=['POST'])
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
@api_bp.route('/station/feature_search', methods=['POST'])
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

def hyperparameter_feature_search_method(request_data):
    hp_req = HyperParamSearchRequest(**request_data)

    # 初始化配置和数据库连接
    config = Config()
    db = DataBase(**config.database)

    # 1. 获取历史负荷数据
    load_query = """
                SELECT load_time, load_data
                FROM ustlf_station_history_load
                WHERE site_id = '{}' AND load_time <= '{}'
                ORDER BY load_time
            """.format(hp_req.site_id, hp_req.end_date)
    load_df = db.query(load_query)

    # 2. 获取历史气象数据
    meteo_query = """
                SELECT meteo_times, relative_humidity_2m, surface_pressure,
                       precipitation, wind_speed_10m, temperature_2m, shortwave_radiation
                FROM ustlf_station_meteo_data
                WHERE site_id = '{}' AND meteo_times<='{}'
                ORDER BY meteo_times
            """.format(hp_req.site_id, hp_req.end_date)
    meteo_df = db.query(meteo_query)

    # 3. 特征工程处理
    h = 16
    # processor = DataProcessor()
    feature_engine = FeatureEngineer()

    # 生成所有可能的特征组合
    # config = Config()
    best_features = config.config['base_features']

    df, all_features = feature_engine.extract_features(load_df, meteo_df, h=h)
    # all_features = list(base_features.columns)
    # all_features.remove('load_data')
    df = df[96 * 7:-96]

    # 4. 生成超参数搜索空间

    param_grid = {
        'boosting_type': ['gbdt', 'dart'],
        'objective': ['rmse', 'mae'],
        'n_jobs': [4],
        'max_depth': [5, 10, -1],
        'subsample': [0.5, 0.7, 0.9],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_samples': [10, 20, 30],
        'n_estimators': [500, 1000, 1500],
        'boost_from_average': [False, True],
        'random_state': [42],
        'verbosity': [-1]
    }
    # 5. 数据集分割
    all_features = set(all_features)
    all_features = list(all_features)
    all_features.sort()
    X = df[all_features]
    y = df['load_data']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 6. 特征和超参数联合搜索

    # 限制搜索次数 如果超参数搜索次数连续超过max_search没有更好的模型就会停止搜索
    max_search = 5
    search_count = 0

    # 基准params： 以往经验中得到的最好的超参数，如果超过50次没有对精度提升，则使用基准params
    # best_params = config.config['model_params']['lightgbm']
    best_params = {'boost_from_average': True, 'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_depth': 10,
                   'min_child_samples': 10, 'n_estimators': 1500, 'n_jobs': 4, 'objective': 'mae', 'random_state': 42,
                   'subsample': 0.5, 'verbosity': -1}
    model = ML_Model(
        model_name=f'site_{hp_req.site_id}_hp_search',
        model_params=best_params,
        model_path=f"./models_pkl/site_id_{hp_req.site_id}"
    )
    model.get_model()
    model.model_train(X_train, y_train)
    y_pred = model.model_predict(X_val)
    best_score = mean_squared_error(y_val, y_pred)

    for params in tqdm.tqdm(ParameterGrid(param_grid)):
        if search_count >= max_search:
            break
        search_count += 1
        # print(f"Search count: {search_count}")
        # print(f"Current params: {params}")
        model = ML_Model(
            model_name=f'site_{hp_req.site_id}_hp_search',
            model_params=params,
            model_path=f"./models_pkl/site_id_{hp_req.site_id}"
        )
        model.get_model()
        model.model_train(X_train, y_train)
        y_pred = model.model_predict(X_val)
        score = mean_squared_error(y_val, y_pred)
        if score < best_score:
            best_score = score
            best_params = params
            print('精度提升了')
            print(f"Current params: {params}")
            print(f"Current score: {score}")
            print("*" * 100)

    print(f"Best score: {best_score}")
    print(f"Best params: {best_params}")

    best_score_feature_search = float('inf')
    # 在基础输入特征的基础上对其余特征进行遍历，若遍历到的新特征对进度有提升，则纳入基础输入特征
    for feature in tqdm.tqdm(all_features):
        if feature in best_features:
            continue
        input_feature = best_features + [feature]
        X_train_new = X_train[input_feature]
        X_val_new = X_val[input_feature]
        model = ML_Model(
            model_name=f'site_{hp_req.site_id}_hp_search',
            model_params=best_params,
            model_path=f"./models_pkl/site_id_{hp_req.site_id}"
        )
        model.get_model()
        model.model_train(X_train_new, y_train)
        y_pred = model.model_predict(X_val_new)
        score = mean_squared_error(y_val, y_pred)
        if score < best_score_feature_search:
            best_score_feature_search = score
            best_features = input_feature
            print("精度提升了")
            print(f"Current score: {score}")
            print(f"输入特征:{input_feature}")
            print("*" * 100)
    # 如果搜了一圈还是没有全量特征的准确率高，那就继续沿用全量数据
    if best_score_feature_search > best_score:
        best_features = all_features

    # 7. 保存最佳参数组合，暂时不存入

    # # 转化成存入sql的格式
    # feature_info = '[ ' + ' , '.join(f'"{item}"' for item in best_features) + ' ]'
    # params = '{' + ', '.join(
    #     f"''{key}'':{value}" if isinstance(value, int) else f"''{key}'':''{value}''" for key, value in
    #     best_params.items()) + '}'
    #
    # replace_sql = '''
    #         REPLACE INTO ustlf_model_feature_hp_info (site_id, feature_info, hyperparams_info, update_time) VALUES
    #         ('{}', '{}', '{}', '{}');
    #         '''.format(hp_req.site_id, feature_info, params, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # db.execute(replace_sql)

    return jsonify(CommonResponse(
        code="200",
        msg="超参数搜索成功",
        data={"best_params": best_params,
              "best_feature": best_features}
    ).model_dump())

# 5. 超参数&输入特征搜索搜索接口
@api_bp.route('/station/hp_feature_search', methods=['POST'])
def hyperparameter_feature_search():
    """超参数搜索接口"""
    try:
        # 验证请求数据
        data = request.get_json()
        search_res = hyperparameter_feature_search_method(data)
        return search_res

    except Exception as e:
        return jsonify(CommonResponse(
            code="500",
            msg=f"超参数搜索失败: {str(e)}"
        ).model_dump())


def get_history_meteo_method(request_data):
    try:
        meteo_req = HistoryMeteoRequest(**request_data)

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
        end_time = current_time.replace(hour=10, minute=45, second=0, microsecond=0)
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
        # url = "http://8.212.49.208:5009/weather"
        url = "http://192.168.238.149:5009/weather"
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
        data = res['data']
        data = data['par']

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
        logger.info(f"电站{meteo_req.site_id}历史气象数据拉取任务完成")


    except Exception as e:
        return jsonify(CommonResponse(
            code="401",
            msg=f"参数验证失败: {str(e)}"
        ).model_dump())

    return jsonify(CommonResponse(
            code="200",
            msg=f"历史气象拉取成功"
        ).model_dump())

# 6. 历史气象拉取接口
@api_bp.route('/station/get_history_meteo', methods=['POST'])
def get_history_meteo():
    """历史气象拉取接口"""
    request_data = request.get_json()
    try:
        res = get_history_meteo_method(request_data)
        return res
    except Exception as e:
        return jsonify(CommonResponse(
            code="401",
            msg=f"参数验证失败: {str(e)}"
        ).model_dump())

def train_model_method(request_data):
    train_req = ModelTrainRequest(**request_data)

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
    if len(res) == 0:
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
                """.format(train_req.site_id, end_date)
    load_df = db.query(load_query)
    if len(load_df)<90*96:
        logger.warning(f"电站{train_req.site_id}历史负荷数据不足3个月")
        return {
            'code': "401",
            'msg': f"电站{train_req.site_id}历史负荷数据不足3个月"}

    # todo:获取气象负荷
    meteo_query = """
                                SELECT *
                                FROM ustlf_station_meteo_data
                                WHERE site_id = '{}' AND meteo_times <'{}'
                                ORDER BY meteo_times
                            """.format(train_req.site_id, end_date)
    meteo_df = db.query(meteo_query)
    if len(meteo_df)<90*96:
        logger.warning(f"电站{train_req.site_id}气象数据不足3个月")
        return {
            'code':"401",
            'msg':f"电站{train_req.site_id}气象数据不足3个月"}

    # todo: 将气象数据与负荷数据结合，然后根据现在的特征组合，分类出输入特征和标签，
    H_list = [i * 0.25 for i in range(1, 17)]
    df_list = []
    for H in H_list:
        feature_engine = FeatureEngineer()
        df_i, feature_columns = feature_engine.extract_features(load_df, meteo_df, h=H)
        logger.info(f"{train_req.site_id}特征工程完成第{H}小时的训练数据准备，共生成 {len(feature_columns)} 个特征")
        df_i_copy = df_i.copy()
        df_list.append(df_i_copy)
    input_feature = feature_columns
    if feature_info_table:  # 如果取得模型输入参数，更改为从数据库中获取的模型
        input_feature = feature_info_table

    input_feature = list(set(input_feature))
    input_feature.sort()
    # todo: 对list中的每个数据集进行训练，保存模型以电站id为目录名，i表示预测的未来第几个负荷值的模型
    for i, df in enumerate(df_list):
        df = df[:-96]
        X_train, Y_train = df[input_feature], df['load_data']
        model_path = "./models_pkl/site_id_{}".format(train_req.site_id)
        lgb_model = ML_Model(model_name='site_id_{}_model_{}'.format(train_req.site_id, i), model_params=hyperparams,
                             model_path=model_path)
        lgb_model.get_model()
        lgb_model.model_train(X_train, Y_train)
        lgb_model.save_model()
    logger.info(f"电站{train_req.site_id}模型训练成功")
    # 修改station_info表中的trained数据为1
    update_sql = """
                    UPDATE ustlf_station_info
                    SET trained = 1
                    WHERE site_id = '{}'
                """.format(train_req.site_id)

    db.execute(update_sql)

    return {
        'code': "200",
        'msg': "模型训练成功",
        'data': f"电站{train_req.site_id}模型训练成功,模型文件保存在{model_path}"
    }

    # return jsonify(CommonResponse(
    #     code="200",
    #     msg="模型训练成功",
    #     data={"model_path保存到文件中": model_path}
    # ).model_dump())

def test2(data):
    print(data)
    time.sleep(10)
    return data

@api_bp.route('/station/model_test', methods=['POST'])
def test():
    request_data = request.get_json()
    data = request.get_json()
    with Pool(processes=4) as pool:
        res = pool.map(train_model_method, (data))
        # res = train_model_method(data)
        logger.info("模型训练成功")
        return res



# 7. 模型训练接口
@api_bp.route('/station/model_train', methods=['POST'])
def train_model():
    """模型训练接口"""
    logger.info("开始模型训练...")
    try:
        data = request.get_json()
        with Pool(processes=6) as pool:
            res = pool.map(train_model_method, ([data]))
            # res = train_model_method(data)
            logger.info("模型训练成功")
            return res

    except Exception as e:
        logger.error(f"模型训练失败: {str(e)}")
        return jsonify(CommonResponse(
            code="500",
            msg=f"模型训练失败: {str(e)}"
        ).model_dump())


def get_forcast_meteo_method(request_data):
    meteo_req = ForecastMeteoRequest(**request_data)
    # # 获取数据库连接
    logger.info("开始初始化数据库...")
    config = Config()
    # logger.info("配置加载成功")
    # logger.info(f"数据库配置: {config.database}")
    db = DataBase(**config.database)

    # 气象开始时间于结束时间
    start_time = meteo_req.start_time
    end_time = meteo_req.end_time

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
    current_time1 = current_time.strftime('%Y-%m-%d')

    # 调用气象API
    # url = "http://8.212.49.208:5009/weather"
    url = "http://192.168.238.149:5009/weather"
    headers = {'apikey': 'renewable_api_key'}

    params = {
        'latitude': latitude,
        'longitude': longitude,
        'timezone': 'Asia/Shanghai',
        'forecast_days': 4,
        'interval': '15T',
        'start_date': current_time1,     # 此处的日期需要根据实际情况进行修改
        'par': 'shortwave_radiation,temperature_2m,relative_humidity_2m,surface_pressure,precipitation,wind_speed_10m'
    }
    if start_time:
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'timezone': 'Asia/Shanghai',
            'forecast_days': 4,
            'interval': '15T',
            'start_date': start_time,
            'end_date': end_time,
            'par': 'shortwave_radiation,temperature_2m,relative_humidity_2m,surface_pressure,precipitation,wind_speed_10m'
        }


    response = requests.get(url=url, params=params, headers=headers)
    res = response.json()
    data = res['data']
    data = data['par']

    # 处理API返回数据
    temperature = data['temperature_2m']
    pressure = data['surface_pressure']
    humidity = data['relative_humidity_2m']
    irradiation = data['shortwave_radiation']
    precipitation = data['precipitation']
    wind_speed_10m = data['wind_speed_10m']
    timestamp = data['time']

    data_dict = {
        'meteo_times': timestamp,
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

    db.insert(table='ustlf_station_forecast_meteo_data', df=df)
    logger.info(f"电站{meteo_req.site_id}预测气象拉取成功")
    return jsonify(CommonResponse(
        code="200",
        msg=f"预测气象拉取成功"
    ).model_dump())

# 8. 预测气象保存
@api_bp.route('/station/get_forcast_meteo', methods=['POST'])
def get_forcast_meteo():
    """预测气象拉取接口"""
    try:
        data = request.get_json()
        get_forcast_meteo_method(data)
    except Exception as e:
        return jsonify(CommonResponse(
            code="401",
            msg=f"参数验证失败: {str(e)}"
        ).model_dump())

    return jsonify(CommonResponse(
            code="200",
            msg=f"预测气象拉取成功"
        ).model_dump())



# 9.获取电站列表的接口
# @api_bp.route('/station/list', methods=['GET'])
# def get_station_list():
#     """获取电站列表接口"""
#     try:
#         config = Config()
#         db = DataBase(**config.database)
#
#         query = "SELECT site_id FROM ustlf_station_info"
#         sites = db.query(query)
#
#         if sites.empty:
#             return jsonify(CommonResponse(
#                 code="401",
#                 msg="未找到任何电站"
#             ).model_dump())
#
#         return jsonify(CommonResponse(
#             code="200",
#             msg="获取电站列表成功",
#             data={"site_ids": sites['site_id'].tolist()}
#         ).model_dump())
#
#     except Exception as e:
#         return jsonify(CommonResponse(
#             code="500",
#             msg=f"获取电站列表失败: {str(e)}"
#         ).model_dump())

@api_bp.route('/station/list', methods=['GET'])
def get_station_list():
    """获取电站列表接口"""
    try:
        config = Config()
        db = DataBase(**config.database)

        query = "SELECT site_id FROM ustlf_station_info"
        sites = db.query(query)

        if sites.empty:
            return {
                'code':"401",
                'msg':"未找到任何电站"
            }

        return {
            'code':"200",
            'msg' : "获取电站列表成功",
            'data' : sites['site_id'].tolist()
        }


    except Exception as e:

        return {
            'code': "500",
            'msg': f"获取电站列表失败: {str(e)}"
        }


if __name__ == '__main__':
    res = get_station_list()
    print(res)
    print(res['data'])



    response = requests.get('http://127.0.0.1:5000/ustlf/station/list')
    print(response)
    stations = response.json()['data']
    stations_count = len(stations)



# -*- coding: utf-8 -*-
from flask import Flask
import sys
sys.path.append("..")
from utils.config import Config
from utils.scheduler import Scheduler
from utils.database import DataBase
from api.schemas import StationRegisterRequest
from api.routes import get_history_meteo_method, train_model_method

from loguru import logger

logger.add(
    "logs/app_info.log",
    rotation="1 day",
    retention="90 days",
    level="INFO",
    format="{time} {level} {message}",
    enqueue=True,
    colorize=True,
    backtrace=True,
    diagnose=True,
)
# logger.warning()
# logger.add(
#     "logs/app_warning.log",
#     rotation="1 day",
#     retention="90 days",
#     level="warning",
#     format="{time} {level} {message}",
#     enqueue=True,
#     colorize=True,
#     backtrace=True,
#     diagnose=True,
# )

# logger.add(
#     "logs/app_error.log",
#     rotation="1 day",
#     retention="90 days",
#     level="error",
#     format="{time} {level} {message}",
#     enqueue=True,
#     colorize=True,
#     backtrace=True,
#     diagnose=True,
# )


def initialize_stations():
    """初始化电站信息"""
    logger.info("开始初始化电站信息...")
    config = Config()
    db = DataBase(**config.database)

    # 查询电站信息
    query = "SELECT site_id FROM ustlf_station_info"
    stations = db.query(query)
    if stations.empty:
        logger.warning("没有找到任何电站信息，请检查数据库中是否有数据。")
        return

    for site_id in stations['site_id']:
        logger.info(f"处理电站: {site_id}")
        # 获取历史负荷数据
        load_query = f"""
            SELECT load_time,load_data FROM ustlf_station_history_load
            WHERE site_id = '{site_id}'
        """
        load_data = db.query(load_query)

        if load_data.empty:
            logger.warning(f"电站 {site_id} 的历史负荷数据为空，保留 init_success 为 0")
            continue

        time_span = load_data['load_time'].max() - load_data['load_time'].min()
        if time_span.days < 90:
            logger.warning(f"电站 {site_id} 的历史负荷数据时间不足3个月，保留 init_success 为 0")
            continue

        # 调用模型训练接口
        logger.info(f"开始对电站 {site_id} 进行模型训练...")
        # 这里调用模型训练的函数
        # result = get_history_meteo_method(site_id)

        # todo：测试阶段可以先不开启，该训练是初始化训练使用的
        # train_model_method({'site_id': site_id})

    logger.info("电站初始化完成。")


def create_app():
    app = Flask(__name__)

    # 初始化自定义调度器
    # scheduler = Scheduler()

    # 添加气象数据获取任务
    # scheduler.add_meteo_fetch_job()

    # 每天在如下几个时间点更新预测气象数据
    # scheduler.add_future_meteo_job(hour=2)
    # scheduler.add_future_meteo_job(hour=8)
    # scheduler.add_future_meteo_job(hour=14)
    # scheduler.add_future_meteo_job(hour=20)

    # 每日23点对注册电站模型训练
    # scheduler.add_model_training_job(hour=23, minute=15)

    # 超参数寻优在每日1号的10点进行
    # scheduler.add_feature_search_job(day=1, hour=10)

    # initialize_stations()  # 调用初始化函数

    # scheduler.start()

    # 注册蓝本
    # from .routes import api_bp
    from api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/ustlf')

    return app


app = create_app()

if __name__ == '__main__':
    from gevent import pywsgi

    app.debug = True
    # server = pywsgi.WSGIServer(('0.0.0.0', 5555), app)
    # server.serve_forever()

    app.run(host='0.0.0.0', port=5555, threaded=False)

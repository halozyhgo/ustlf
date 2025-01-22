# -*- coding: utf-8 -*-
from flask import Flask
from load_forecast_platform.utils.scheduler import Scheduler
from .routes import get_history_meteo_method

def create_app():
    app = Flask(__name__)
    
    # 初始化自定义调度器
    scheduler = Scheduler()
    
    # 添加气象数据获取任务
    scheduler.add_meteo_fetch_job()

    # 每天在如下几个时间点更新预测气象数据
    scheduler.add_future_meteo_job(hour=2)
    scheduler.add_future_meteo_job(hour=8)
    scheduler.add_future_meteo_job(hour=14)
    scheduler.add_future_meteo_job(hour=20)

    # 每日23点对注册电站模型训练
    scheduler.add_model_training_job(hour=23, minute=15)

    # 超参数寻优在每日1号的10点进行
    scheduler.add_feature_search_job(day=1, hour=10)


    scheduler.start()
    
    # 注册蓝本
    from .routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/ustlf')
    
    return app

app = create_app()

if __name__ == '__main__':
    from gevent import pywsgi
    app.debug=True
    # server = pywsgi.WSGIServer(('192.168.156.222', 5000), app)
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    server.serve_forever()

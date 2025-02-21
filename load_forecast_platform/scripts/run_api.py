# -*- coding: utf-8 -*-
from flask_apscheduler import APScheduler
from api.routes import get_history_meteo, hyperparameter_feature_search
from api import app

def init_scheduler(app):
    scheduler = APScheduler()
    scheduler.init_app(app)
    
    # 每日19:10获取历史气象数据
    @scheduler.task('cron', id='fetch_history_meteo', hour=19, minute=10)
    def fetch_history_meteo_job():
        get_history_meteo()
    
    # 每月1日01:10执行超参数和特征搜索
    @scheduler.task('cron', id='hyperparameter_search', day=1, hour=1, minute=10)
    def hyperparameter_search_job():
        hyperparameter_feature_search()
    
    scheduler.start()

if __name__ == "__main__":
    init_scheduler(app)
    from gevent import pywsgi
    app.debug = True
    # server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    # server.serve_forever()

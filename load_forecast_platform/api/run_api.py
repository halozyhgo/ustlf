# -*- coding: utf-8 -*-
# from api import app
from utils.scheduler import Scheduler
from api.routes import get_history_meteo, hyperparameter_feature_search
from flask import Flask


def init_scheduler():
    scheduler = Scheduler()

    # 每日19点获取历史气象数据
    scheduler.add_model_training_job(
        get_history_meteo,
        hour=19
    )

    # 每日20点执行超参数和特征搜索
    scheduler.add_model_training_job(
        hyperparameter_feature_search,
        hour=20
    )

    scheduler.start()

app = Flask(__name__)
if __name__ == "__main__":
    init_scheduler()
    from gevent import pywsgi

    # app.debug = True
    # server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    # server.serve_forever()

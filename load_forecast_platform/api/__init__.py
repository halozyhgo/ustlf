# -*- coding: utf-8 -*-
from flask import Flask
from load_forecast_platform.utils.scheduler import Scheduler
from .routes import get_history_meteo

def create_app():
    app = Flask(__name__)
    
    # 初始化自定义调度器
    scheduler = Scheduler()
    
    # 添加气象数据获取任务
    scheduler.add_meteo_fetch_job(lambda: get_history_meteo())
    scheduler.start()
    
    # 注册蓝本
    from .routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/ustlf')
    
    return app

app = create_app()

if __name__ == '__main__':
    from gevent import pywsgi
    app.debug=True
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    server.serve_forever()

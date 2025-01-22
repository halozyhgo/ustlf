# -*- coding: utf-8 -*-
from flask import Flask
from api import app
from scheduler.scheduler import init_scheduler
from loguru import logger

def create_app():
    """创建并配置应用"""
    try:
        # 初始化调度器
        scheduler = init_scheduler()
        logger.info("定时任务调度器初始化成功")
        return app
    except Exception as e:
        logger.error(f"应用初始化失败: {str(e)}")
        raise

if __name__ == '__main__':
    from gevent import pywsgi
    app = create_app()
    # app.run(host='0.0.0.0', port=5000)
    app.debug = True
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    server.serve_forever()
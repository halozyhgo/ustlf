# -*- coding: utf-8 -*-
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger
from datetime import datetime
from load_forecast_platform.api.routes import get_history_meteo_method,get_forcast_meteo_method,train_model_method,hyperparameter_feature_search_method
import pytz

class Scheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Shanghai'))
        self.jobs = {}

    def add_feature_search_job(self, day=1, hour=10):
        """添加特征搜索任务"""
        def search_features_for_stations():
            from load_forecast_platform.api import app
            import requests
            
            with app.app_context():
                try:
                    # 获取所有电站列表
                    response = requests.get('http://localhost:5555/ustlf/station/list')
                    if response.status_code != 200:
                        logger.error(f"获取电站列表失败: {response.status_code}")
                        return
                        
                    stations = response.json()['data']
                    station_count = len(stations)
                    
                    for station in stations:
                        try:
                            # 为每个电站创建特征搜索请求
                            request_data = {
                                'site_id': station,
                                # 'start_time': '...',  # 需要根据实际需求设置时间
                                # 'end_date': '2024-09-01'
                            }
                            hy_search_res = hyperparameter_feature_search_method(request_data)
                        except Exception as e:
                            logger.error(f"电站 {station['id']} 特征搜索失败: {str(e)}")
                            
                    logger.info(f"成功处理 {station_count} 个电站的特征搜索")
                except Exception as e:
                    logger.error(f"特征搜索任务执行失败: {str(e)}")

        job_id = 'feature_search'
        if job_id in self.jobs:
            self.remove_job(job_id)

        trigger = CronTrigger(day=day, hour=hour)
        # trigger = CronTrigger(second=1)
        job = self.scheduler.add_job(
            search_features_for_stations,
            trigger=trigger,
            id=job_id
        )
        self.jobs[job_id] = job
        logger.info(f"特征搜索任务已添加，将在每月{day}日{hour}点执行")

    def add_model_training_job(self, hour=1, minute=15):
        """添加模型训练任务"""
        def train_models_for_stations():
            from load_forecast_platform.api import app
            import requests
            
            with app.app_context():
                try:
                    # 获取所有电站列表
                    response = requests.get('http://127.0.0.1:5000/ustlf/station/list')
                    if response.status_code != 200:
                        logger.error(f"获取电站列表失败: {response.status_code}")
                        return
                        
                    stations = response.json()['data']
                    station_count = len(stations)
                    
                    for station in stations:
                        try:
                            # 为每个电站创建训练请求
                            request_data = {
                                'site_id': station,
                                # 'end_date': '2024-09-01 23:00'  # 需要根据实际需求设置时间 目前是测试使用的时间
                            }
                            res = train_model_method(request_data)
                        except Exception as e:
                            logger.error(f"电站 {station['id']} 模型训练失败: {str(e)}")
                            
                    logger.info(f"成功处理 {station_count} 个电站的模型训练")
                except Exception as e:
                    logger.error(f"模型训练任务执行失败: {str(e)}")

        job_id = 'model_training'
        if job_id in self.jobs:
            self.remove_job(job_id)

        trigger = CronTrigger(hour=hour, minute=minute)
        # trigger = CronTrigger(second=1)
        job = self.scheduler.add_job(
            train_models_for_stations,
            trigger=trigger,
            id=job_id
        )
        self.jobs[job_id] = job
        logger.info(f"模型训练任务已添加，将在每日{hour}:{minute}执行")

    def add_meteo_fetch_job(self, hour=19):
        """添加气象数据获取任务"""
        def fetch_meteo_for_stations():
            from load_forecast_platform.api import app
            import requests
            
            with app.app_context():
                try:
                    # 获取所有电站列表
                    response = requests.get('http://127.0.0.1:5000/ustlf/station/list')
                    if response.status_code != 200:
                        logger.error(f"获取电站列表失败: {response.status_code}")
                        return
                        
                    stations = response.json()['data']
                    station_count = len(stations)
                    
                    for station in stations:
                        try:
                            # 为每个电站创建请求数据
                            print("获取下面电站的气象数据")
                            print(station)
                            request_data = {
                                'site_id': station,
                                # 'start_time': '...',  # 需要根据实际需求设置时间
                                # 'end_time': '...'
                            }
                            res = get_history_meteo_method(request_data)
                            print(res)
                            # response = requests.post('http://127.0.0.1:5000/ustlf/station/get_history_meteo', params=request_data)
                        except Exception as e:
                            logger.error(f"电站 {station['id']} 气象数据获取失败: {str(e)}")
                            
                    logger.info(f"成功处理 {station_count} 个电站的气象数据")
                except Exception as e:
                    logger.error(f"气象数据获取任务执行失败: {str(e)}")

        job_id = 'meteo_fetch'
        if job_id in self.jobs:
            self.remove_job(job_id)

        trigger = CronTrigger(hour=19,minute=00)
        # trigger = CronTrigger(second=1,)
        job = self.scheduler.add_job(
            fetch_meteo_for_stations,
            trigger=trigger,
            id=job_id
        )
        self.jobs[job_id] = job
        logger.info(f"气象数据获取任务已添加，将在每日{hour}点执行")

    def add_future_meteo_job(self, hour=19):
        """获取预测气象数据获取任务"""

        def get_forecast_meteo_for_stations():
            from load_forecast_platform.api import app
            import requests

            with app.app_context():
                try:
                    # 获取所有电站列表
                    response = requests.get('http://127.0.0.1:5000/ustlf/station/list')
                    if response.status_code != 200:
                        logger.error(f"获取电站列表失败: {response.status_code}")
                        return

                    stations = response.json()['data']
                    station_count = len(stations)

                    for station in stations:
                        try:
                            # 为每个电站创建请求数据
                            print("获取下面电站的气象数据")
                            print(station)
                            request_data = {
                                'site_id': station,
                                # 'start_time': '...',  # 需要根据实际需求设置时间
                                # 'end_time': '...'
                            }
                            res = get_forcast_meteo_method(request_data)
                            print(res)
                            # response = requests.post('http://127.0.0.1:5000/ustlf/station/get_history_meteo', params=request_data)
                        except Exception as e:
                            logger.error(f"电站 {station['id']} 气象数据获取失败: {str(e)}")

                    logger.info(f"成功处理 {station_count} 个电站的气象数据")
                except Exception as e:
                    logger.error(f"气象数据获取任务执行失败: {str(e)}")

        job_id = 'meteo_fetch'
        if job_id in self.jobs:
            self.remove_job(job_id)

        trigger = CronTrigger(hour=hour,minute=1)
        # trigger = CronTrigger(second=1,)
        job = self.scheduler.add_job(
            get_forecast_meteo_for_stations,
            trigger=trigger,
            id=job_id
        )
        self.jobs[job_id] = job
        logger.info(f"气象数据获取任务已添加，将在每日{hour}点执行")

    def remove_job(self, job_id):
        """移除定时任务"""
        if job_id in self.jobs:
            self.scheduler.remove_job(job_id)
            del self.jobs[job_id]
            logger.info(f"已移除任务: {job_id}")

    def start(self):
        """启动调度器"""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("调度器已启动")

    def shutdown(self):
        """关闭调度器"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("调度器已关闭")

    def list_jobs(self):
        """列出所有定时任务"""
        return self.scheduler.get_jobs()

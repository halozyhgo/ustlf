# -*- coding: utf-8 -*-
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger
from datetime import datetime
import pytz

class Scheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Shanghai'))
        self.jobs = {}

    def add_feature_search_job(self, func, day=1, hour=10):
        """添加特征搜索任务"""
        job_id = 'feature_search'
        if job_id in self.jobs:
            self.remove_job(job_id)

        trigger = CronTrigger(day=day, hour=hour)
        job = self.scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id
        )
        self.jobs[job_id] = job
        logger.info(f"特征搜索任务已添加，将在每月{day}日{hour}点执行")

    def add_model_training_job(self, func, hour=20):
        """添加模型训练任务"""
        job_id = 'model_training'
        if job_id in self.jobs:
            self.remove_job(job_id)

        trigger = CronTrigger(hour=hour)
        job = self.scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id
        )
        self.jobs[job_id] = job
        logger.info(f"模型训练任务已添加，将在每日{hour}点执行")

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

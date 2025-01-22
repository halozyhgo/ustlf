# -*- coding: utf-8 -*-
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger
from .tasks import fetch_history_meteo, search_hyperparameters

def init_scheduler():
    """初始化调度器"""
    try:
        scheduler = BackgroundScheduler()
        
        # 添加每日19点拉取历史气象任务
        scheduler.add_job(
            fetch_history_meteo,
            CronTrigger(hour=19, minute=0),
            id='fetch_history_meteo',
            name='每日19点拉取历史气象数据'
        )
        
        # 添加每月1日10点进行超参数搜索任务
        scheduler.add_job(
            search_hyperparameters,
            CronTrigger(day=1, hour=10, minute=0),
            id='search_hyperparameters',
            name='每月1日10点进行超参数搜索'
        )
        
        scheduler.start()
        logger.info("定时任务调度器已启动")
        
        return scheduler
        
    except Exception as e:
        logger.error(f"初始化调度器失败: {str(e)}")
        raise 
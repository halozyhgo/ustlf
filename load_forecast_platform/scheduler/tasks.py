# -*- coding: utf-8 -*-
import requests
from datetime import datetime, timedelta
from loguru import logger
from utils.config import Config
from utils.database import DataBase

def get_site_ids():
    """获取所有电站ID"""
    try:
        db = DataBase(**Config().database)
        query = "SELECT site_id FROM ustlf_station_info"
        sites = db.query(query)
        return sites['site_id'].tolist() if not sites.empty else []
    except Exception as e:
        logger.error(f"获取电站列表失败: {str(e)}")
        return []

def fetch_history_meteo():
    """每日19点拉取历史气象数据任务"""
    try:
        logger.info("开始执行历史气象数据拉取任务")
        
        # 1. 获取所有电站ID
        site_ids = get_site_ids()
        if not site_ids:
            logger.warning("未找到任何电站")
            return
            
        # 2. 计算时间范围
        end_time = datetime.now().replace(hour=19, minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(days=3)
        
        # 3. 为每个电站拉取气象数据
        for site_id in site_ids:
            try:
                data = {
                    "site_id": site_id,
                    "meteo_id": 1,
                    "start_time": start_time.strftime("%Y-%m-%d"),
                    "end_time": end_time.strftime("%Y-%m-%d")
                }
                
                response = requests.post(
                    "http://localhost:5000/ustlf/station/get_history_meteo", 
                    json=data
                )
                result = response.json()
                
                if result['code'] == "200":
                    logger.info(f"电站 {site_id} 的气象数据拉取成功")
                else:
                    logger.error(f"电站 {site_id} 的气象数据拉取失败: {result['msg']}")
                    
            except Exception as e:
                logger.error(f"处理电站 {site_id} 的气象数据时出错: {str(e)}")
                continue
                
        logger.info("历史气象数据拉取任务完成")
        
    except Exception as e:
        logger.error(f"历史气象数据拉取任务失败: {str(e)}")

def search_hyperparameters():
    """每月1日10点进行超参数搜索任务"""
    try:
        logger.info("开始执行超参数搜索任务")
        
        # 1. 获取所有电站ID
        site_ids = get_site_ids()
        if not site_ids:
            logger.warning("未找到任何电站")
            return
            
        # 2. 为每个电站执行超参数搜索
        for site_id in site_ids:
            try:
                data = {
                    "site_id": site_id,
                    "end_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                response = requests.post(
                    "http://localhost:5000/ustlf/station/hp_feature_search", 
                    json=data
                )
                result = response.json()
                
                if result['code'] == "200":
                    logger.info(f"电站 {site_id} 的超参数搜索成功")
                else:
                    logger.error(f"电站 {site_id} 的超参数搜索失败: {result['msg']}")
                    
            except Exception as e:
                logger.error(f"处理电站 {site_id} 的超参数搜索时出错: {str(e)}")
                continue
                
        logger.info("超参数搜索任务完成")
        
    except Exception as e:
        logger.error(f"超参数搜索任务失败: {str(e)}") 
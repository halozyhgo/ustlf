# -*- coding: utf-8 -*-
import logging
import os
from datetime import datetime
from utils.database import DataBase
from utils.config import Config
from loguru import logger
import pandas as pd

class DBHandler(logging.Handler):
    """自定义数据库日志处理器"""
    def __init__(self, site_id=None):
        super().__init__()
        self.site_id = site_id
        config = Config().database
        self.db = DataBase(**config)
        
    def emit(self, record):
        """将日志写入数据库"""
        try:
            # 格式化日志消息
            log_entry = self.format(record)
            
            # 构建日志数据
            log_data = pd.DataFrame([{
                'site_id': self.site_id,
                'info': log_entry,
                'date': datetime.now()
            }])
            
            # 使用insert方法插入数据
            self.db.insert('ustlf_log_info', log_data)
                
        except Exception as e:
            # 如果写入数据库失败，打印到控制台
            print(f"Error writing to database: {str(e)}")
            print(f"Log message: {log_entry}")

def setup_logger(name, site_id=None, log_file=None, level=logging.INFO):
    """设置日志记录器
    
    Args:
        name: 日志记录器名称
        site_id: 电站ID，用于写入数据库
        log_file: 日志文件路径，如果为None则输出到控制台
        level: 日志级别
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 如果没有处理器，添加处理器
    if not logger.handlers:
        # 添加数据库处理器
        if site_id is not None:
            db_handler = DBHandler(site_id)
            db_handler.setFormatter(formatter)
            logger.addHandler(db_handler)
            
        if log_file:
            # 确保日志目录存在
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            # 创建文件处理器
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        else:
            # 创建控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    
    return logger 
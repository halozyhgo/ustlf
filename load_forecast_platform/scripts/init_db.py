# -*- coding: utf-8 -*-
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.config import Config
from utils.db_utils import DatabaseConnection
from sqlalchemy import text

def create_tables(db):
    """创建必要的数据表"""
    
    # 创建电站信息表
    power_station_table = """
    CREATE TABLE IF NOT EXISTS power_station (
        SiteId INT PRIMARY KEY COMMENT '电站id',
        SiteName VARCHAR(255) COMMENT '电站名称',
        Longitude DECIMAL(9,6) COMMENT '经度',
        Latitude DECIMAL(9,6) COMMENT '纬度',
        Stype INT COMMENT '电站类型: 1-光储电站，2-储能电站，3-其他电站',
        RatedCapacity DECIMAL(10,2) COMMENT '额定容量',
        RatedPower DECIMAL(10,2) COMMENT '额定功率',
        RatedPowerPV DECIMAL(10,2) COMMENT '额定光伏发电功率',
        FrequencyLoad INT COMMENT '负荷数据时间分辨率',
        FrequencyMeteo INT COMMENT '气象数据分辨率',
        FirstLoadTime DATETIME COMMENT '负荷开始时间',
        UploadTime DATETIME COMMENT '电站注册时间'
    ) COMMENT='电站信息表';
    """
    
    # 创建历史负荷表
    station_history_load_table = """
    CREATE TABLE IF NOT EXISTS stationhistory_load (
        SiteId INT COMMENT '电站id',
        SiteName VARCHAR(255) COMMENT '电站名称',
        LoadTimeStamp DATETIME COMMENT '负荷时间',
        LoadData DECIMAL(10,2) COMMENT '负荷数值',
        UploadTime DATETIME COMMENT '上传时间',
        PRIMARY KEY (SiteId, LoadTimeStamp)
    ) COMMENT='电站历史负荷表';
    """
    
    try:
        # 测试数据库连接
        print(f"正在连接数据库: {db.db_config['url']}")
        engine = db.engine
        print("数据库连接成功")
        
        # 执行建表语句
        with engine.begin() as conn:
            print("正在创建power_station表...")
            conn.execute(text(power_station_table))
            print("power_station表创建成功")
            
            print("正在创建stationhistory_load表...")
            conn.execute(text(station_history_load_table))
            print("stationhistory_load表创建成功")
            
        print("所有数据表创建成功")
        
    except Exception as e:
        print(f"创建数据表失败: {str(e)}")
        raise
    finally:
        db.close()

if __name__ == '__main__':
    try:
        print("开始初始化数据库...")
        config = Config()
        print("配置加载成功")
        print(f"数据库配置: {config.database}")
        
        db = DatabaseConnection(config.database)
        create_tables(db)
        
    except Exception as e:
        print(f"程序执行失败: {str(e)}")
        sys.exit(1)
# -*- coding: utf-8 -*-
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 修改导入语句，添加项目包前缀
from load_forecast_platform.utils.config import Config
from load_forecast_platform.utils.db_utils import DatabaseConnection
from sqlalchemy import text


def create_tables(db):
    """创建必要的数据表"""

    # 创建电站信息表
    station_info_table = """
    CREATE TABLE IF NOT EXISTS ustlf_station_info (
        Site_Id INT PRIMARY KEY,             -- 电站id（主键，不可为空）
        Site_Name VARCHAR(255) NOT NULL,     -- 电站名称（不可为空）
        Longitude DECIMAL(10, 6) NOT NULL,   -- 经度（不可为空）
        Latitude DECIMAL(10, 6) NOT NULL,    -- 纬度（不可为空）
        Stype TINYINT NOT NULL,              -- 电站类型：1: 光储电站, 2: 储能电站, 3: 其他电站（不可为空）
        Rated_Capacity DECIMAL(10, 2),       -- 额定容量（可为空）
        Rated_Power DECIMAL(10, 2),          -- 额定功率（可为空）
        Rated_Power_PV DECIMAL(10, 2),       -- 额定光伏发电功率（可为空）
        Frequency_Load INT,                  -- 负荷数据时间分辨率（可为空）
        Frequency_Meteo INT,                 -- 气象数据时间分辨率（可为空）
        First_Load_Time DATETIME NOT NULL,   -- 负荷开始时间（不可为空）
        Upload_Time DATETIME NOT NULL        -- 电站注册时间（不可为空）
    );
    """

    # 创建历史负荷表
    station_history_load_table = """
    CREATE TABLE IF NOT EXISTS ustlf_station_history_load (
        Site_Id INT NOT NULL,              -- 电站id（不可为空，外键）
        Site_Name VARCHAR(255) NOT NULL,   -- 电站名称（不可为空）
        Load_TimeStamp DATETIME NOT NULL,  -- 负荷时间（不可为空）
        Load_Data DECIMAL(10, 2) NOT NULL, -- 负荷数值（不可为空）
        Upload_Time DATETIME NOT NULL,     -- 上传时间（不可为空）
        PRIMARY KEY (Site_Id, Load_TimeStamp) -- 联合主键：电站id + 负荷时间
    );
    """

    # 创建气象信息表
    meteo_info_table = """
    CREATE TABLE IF NOT EXISTS ustlf_meteo_info (
        Meteo_Id INT PRIMARY KEY,         -- 气象id（主键）
        Meteo_Name VARCHAR(255) NOT NULL, -- 气象名称（不可为空）
        Upload_Time DATETIME NOT NULL     -- 气象源注册时间（不可为空）
    );
    """

    # 创建电站气象关联表
    station_meteo_mapping_table = """
    CREATE TABLE IF NOT EXISTS ustlf_station_meteo_mapping (
        Site_Id INT NOT NULL,            -- 电站id（不可为空，联合主键）
        Meteo_Id INT NOT NULL,           -- 气象id（不可为空，联合主键）
        Update_Time DATETIME NOT NULL,   -- 更新时间（不可为空）
        PRIMARY KEY (Site_Id, Meteo_Id)  -- 联合主键：电站id和气象id
    );
    """

    # 创建电站气象数据表
    station_meteo_data_table = """
    CREATE TABLE IF NOT EXISTS ustlf_station_meteo_data (
        Site_Id INT NOT NULL,                               -- 电站id（不可为空，联合主键）
        Meteo_Id INT NOT NULL,                              -- 气象id（不可为空，联合主键）
        Meteo_times DATETIME NOT NULL,                      -- 时间戳（不可为空，联合主键）
        Update_Time DATETIME NOT NULL,                      -- 更新时间（不可为空）
        relative_humidity_2m DECIMAL(5, 2) NOT NULL,        -- 2m相对湿度（不可为空，支持两位小数）
        surface_pressure DECIMAL(10, 2) NOT NULL,           -- 气压（不可为空，支持两位小数）
        precipitation DECIMAL(10, 2) NOT NULL,              -- 降水量（不可为空，支持两位小数）
        wind_speed_10m DECIMAL(5, 2) NOT NULL,              -- 10m高度风速（不可为空，支持两位小数）
        temperation_2m DECIMAL(5, 2) NOT NULL,              -- 2m高度温度（不可为空，支持两位小数）
        shortwave_radiation DECIMAL(10, 2),                 -- 向下短波辐照（不可为空，支持两位小数）
        PRIMARY KEY (Site_Id, Meteo_Id, Meteo_times)        -- 联合主键：电站id + 气象id + 时间戳
    );
    """

    # 创建预测结果表
    pred_res_table = """
    CREATE TABLE IF NOT EXISTS ustlf_pred_res (
        Site_Id INT NOT NULL,              -- 电站id
        Meteo_Id INT NOT NULL,             -- 气象id
        Cal_Time DATETIME NOT NULL,        -- 计算时间
        Forcast_Time_Start DATETIME NOT NULL, -- 预测结果的起始时间
        Res_Data DECIMAL(10, 2) NOT NULL,           -- 超短期负荷预测结果
        PRIMARY KEY (Site_Id, Meteo_Id, Cal_Time, Forcast_Time_Start) -- 联合主键
    );
    """

    # 创建日志信息表
    log_info_table = """
    CREATE TABLE IF NOT EXISTS ustlf_log_info (
        Log_id INT AUTO_INCREMENT PRIMARY KEY,  -- 记录id，主键，自增长
        Site_Id INT NOT NULL,                   -- 电站id
        Info TEXT NOT NULL,                     -- 日志信息
        Date DATETIME NOT NULL                  -- 日志录入时间
    );
    """

    # 创建特征超参数信息表
    model_feature_hp_info_table = """
    CREATE TABLE IF NOT EXISTS ustlf_model_feature_hp_info (
        Site_Id INT PRIMARY KEY,              -- 电站id，主键
        Feature_info TEXT NOT NULL,            -- 输入特征信息
        Hyperparams_info TEXT NOT NULL         -- 输入超参数信息
    );
    """

    try:
        # 测试数据库连接
        print(f"正在连接数据库: {db.db_config['url']}")
        engine = db.engine
        print("数据库连接成功")

        # 执行建表语句
        with engine.begin() as conn:
            print("正在创建数据表...")

            tables = [
                ('ustlf_station_info', station_info_table),
                ('ustlf_station_history_load', station_history_load_table),
                ('ustlf_meteo_info', meteo_info_table),
                ('ustlf_station_meteo_mapping', station_meteo_mapping_table),
                ('ustlf_station_meteo_data', station_meteo_data_table),
                ('ustlf_pred_res', pred_res_table),
                ('ustlf_log_info', log_info_table),
                ('ustlf_model_feature_hp_info', model_feature_hp_info_table)
            ]

            for table_name, create_sql in tables:
                print(f"正在创建{table_name}表...")
                conn.execute(text(create_sql))
                print(f"{table_name}表创建成功")

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
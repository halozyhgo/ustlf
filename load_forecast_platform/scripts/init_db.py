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
    create table if not exists ustlf_station_info (
        site_id int primary key,             -- 电站id（主键，不可为空）
        site_name varchar(255) not null,     -- 电站名称（不可为空）
        longitude decimal(10, 6) not null,   -- 经度（不可为空）
        latitude decimal(10, 6) not null,    -- 纬度（不可为空）
        stype tinyint not null,              -- 电站类型：1: 光储电站, 2: 储能电站, 3: 其他电站（不可为空）
        rated_capacity decimal(10, 2),       -- 额定容量（可为空）
        rated_power decimal(10, 2),          -- 额定功率（可为空）
        rated_power_pv decimal(10, 2),       -- 额定光伏发电功率（可为空）
        frequency_load int,                  -- 负荷数据时间分辨率（可为空）
        frequency_meteo int,                 -- 气象数据时间分辨率（可为空）
        first_load_time datetime not null,   -- 负荷开始时间（不可为空）
        upload_time datetime not null        -- 电站注册时间（不可为空）
    );
    """

    # 创建历史负荷表
    station_history_load_table = """
    create table if not exists ustlf_station_history_load (
        site_id int not null,              -- 电站id（不可为空，外键）
        site_name varchar(255) not null,   -- 电站名称（不可为空）
        load_timestamp datetime not null,  -- 负荷时间（不可为空）
        load_data decimal(10, 2) not null, -- 负荷数值（不可为空）
        upload_time datetime not null,     -- 上传时间（不可为空）
        primary key (site_id, load_timestamp) -- 联合主键：电站id + 负荷时间
    );
    """

    # 创建气象信息表
    meteo_info_table = """
    create table if not exists ustlf_meteo_info (
        meteo_id int primary key,         -- 气象id（主键）
        meteo_name varchar(255) not null, -- 气象名称（不可为空）
        upload_time datetime not null     -- 气象源注册时间（不可为空）
    );
    """

    # 创建电站气象关联表
    station_meteo_mapping_table = """
    create table if not exists ustlf_station_meteo_mapping (
        site_id int not null,            -- 电站id（不可为空，联合主键）
        meteo_id int not null,           -- 气象id（不可为空，联合主键）
        update_time datetime not null,   -- 更新时间（不可为空）
        primary key (site_id, meteo_id)  -- 联合主键：电站id和气象id
    );
    """

    # 创建电站气象数据表
    station_meteo_data_table = """
    create table if not exists ustlf_station_meteo_data (
        site_id int not null,                               -- 电站id（不可为空，联合主键）
        meteo_id int not null,                              -- 气象id（不可为空，联合主键）
        meteo_times datetime not null,                      -- 时间戳（不可为空，联合主键）
        update_time datetime not null,                      -- 更新时间（不可为空）
        relative_humidity_2m decimal(5, 2) not null,        -- 2m相对湿度（不可为空，支持两位小数）
        surface_pressure decimal(10, 2) not null,           -- 气压（不可为空，支持两位小数）
        precipitation decimal(10, 2) not null,              -- 降水量（不可为空，支持两位小数）
        wind_speed_10m decimal(5, 2) not null,              -- 10m高度风速（不可为空，支持两位小数）
        temperation_2m decimal(5, 2) not null,              -- 2m高度温度（不可为空，支持两位小数）
        shortwave_radiation decimal(10, 2),                 -- 向下短波辐照（不可为空，支持两位小数）
        primary key (site_id, meteo_id, meteo_times)        -- 联合主键：电站id + 气象id + 时间戳
    );
    """

    # 创建预测结果表
    pred_res_table = """
    create table if not exists ustlf_pred_res (
        site_id int not null,              -- 电站id
        meteo_id int not null,             -- 气象id
        cal_time datetime not null,        -- 计算时间
        forcast_time_start datetime not null, -- 预测结果的起始时间
        res_data decimal(10, 2) not null,           -- 超短期负荷预测结果
        primary key (site_id, meteo_id, cal_time, forcast_time_start) -- 联合主键
    );
    """

    # 创建日志信息表
    log_info_table = """
    create table if not exists ustlf_log_info (
        log_id int auto_increment primary key,  -- 记录id，主键，自增长
        site_id int not null,                   -- 电站id
        info text not null,                     -- 日志信息
        date datetime not null                  -- 日志录入时间
    );
    """

    # 创建特征超参数信息表
    model_feature_hp_info_table = """
    create table if not exists ustlf_model_feature_hp_info (
        site_id int primary key,              -- 电站id，主键
        feature_info text not null,            -- 输入特征信息
        hyperparams_info text not null         -- 输入超参数信息
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
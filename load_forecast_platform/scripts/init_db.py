# -*- coding: utf-8 -*-
import os
import sys
from loguru import logger

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from load_forecast_platform.utils.config import Config
from load_forecast_platform.utils.database import DataBase

def create_tables(db):
    """创建必要的数据表"""

    # 创建电站信息表
    station_info_table = """
            create table if not exists ustlf_station_info (
            idx bigint auto_increment primary key,  -- 自增主键（BIGINT）
            site_id varchar(50) not null unique,         -- 电站id（不可为空，唯一）
            site_name varchar(255) not null,     -- 电站名称（不可为空）
            longitude decimal(10, 6) not null,   -- 经度（不可为空）
            latitude decimal(10, 6) not null,    -- 纬度（不可为空）
            stype tinyint not null,              -- 电站类型：1: 光储电站, 2: 储能电站, 3: 其他电站（不可为空）
            rated_capacity decimal(10, 2),       -- 额定容量（可为空）
            rated_power decimal(10, 2),          -- 额定功率（可为空）
            rated_power_pv decimal(10, 2),       -- 额定光伏发电功率（可为空）
            rated_transform_power decimal(10, 2),-- 变压器额定容量（不可为空）
            first_load_time datetime not null,   -- 负荷开始时间（不可为空）
            upload_time datetime not null,       -- 电站注册时间（不可为空）
            init_success TINYINT(1) DEFAULT 0,   -- 历史负荷长度是否足够3个月
            trained TINYINT(1) DEFAULT 0         -- 是否已经有训练模型
        );
        """

    # 创建历史负荷表
    station_history_load_table = """
            create table if not exists ustlf_station_history_load (
            idx bigint auto_increment primary key,  -- 自增主键（BIGINT）
            site_id varchar(50) not null,                -- 电站id（不可为空）
            load_time datetime not null, -- 负荷时间（不可为空，唯一）
            load_data decimal(10, 2) not null,   -- 负荷数值（不可为空）
            upload_time datetime not null,       -- 上传时间（不可为空）
            unique key (site_id, load_time) -- 联合唯一约束
            
        );
        """

    # 创建气象信息表
    meteo_info_table = """
            create table if not exists ustlf_meteo_info (
            idx int auto_increment primary key,  -- 自增主键
            meteo_id int not null unique,        -- 气象id（不可为空，唯一）
            meteo_name varchar(255) not null,    -- 气象名称（不可为空）
            upload_time datetime not null        -- 气象源注册时间（不可为空）
        );
        """

    # 创建电站气象关联表
    station_meteo_mapping_table = """
            create table if not exists ustlf_station_meteo_mapping (
            idx int auto_increment primary key,  -- 自增主键
            site_id varchar(50) not null,         -- 电站id（不可为空，唯一）
            meteo_id int not null,        -- 气象id（不可为空，唯一）
            update_time datetime not null,        -- 更新时间（不可为空）
            unique key (site_id, meteo_id) -- 联合唯一约束
        );
        """

    # 创建历史电站气象数据表
    station_meteo_data_table = """
        create table if not exists ustlf_station_meteo_data (
            idx bigint auto_increment primary key,  -- 自增主键（BIGINT）
            site_id varchar(50) not null,           -- 电站id（不可为空，唯一）
            meteo_id int not null,                  -- 气象id（不可为空，唯一）
            meteo_times datetime not null,          -- 时间戳（不可为空，唯一）
            update_time datetime not null,          -- 更新时间（不可为空）
            relative_humidity_2m decimal(5, 2) not null,        -- 2m相对湿度（不可为空，支持两位小数）
            surface_pressure decimal(10, 2) not null,           -- 气压（不可为空，支持两位小数）
            precipitation decimal(10, 2) not null,              -- 降水量（不可为空，支持两位小数）
            wind_speed_10m decimal(5, 2) not null,              -- 10m高度风速（不可为空，支持两位小数）
            temperature_2m decimal(5, 2) not null,              -- 2m高度温度（不可为空，支持两位小数）
            shortwave_radiation decimal(10, 2),                 -- 向下短波辐照（可为空，支持两位小数）
            unique key (site_id, meteo_id,meteo_times) -- 联合唯一约束
        );
        """
    # 创建电站各个更新点对应的预测气象数据表
    station_meteo_forecast_data_table = """
            create table if not exists ustlf_station_forecast_meteo_data (
                idx bigint auto_increment primary key,  -- 自增主键（BIGINT）
                site_id varchar(50) not null,           -- 电站id（不可为空，唯一）
                meteo_id int not null,                  -- 气象id（不可为空，唯一）
                meteo_times datetime not null,          -- 时间戳（不可为空，唯一）
                update_time datetime not null,          -- 更新时间（不可为空）
                relative_humidity_2m decimal(5, 2) not null,        -- 2m相对湿度（不可为空，支持两位小数）
                surface_pressure decimal(10, 2) not null,           -- 气压（不可为空，支持两位小数）
                precipitation decimal(10, 2) not null,              -- 降水量（不可为空，支持两位小数）
                wind_speed_10m decimal(5, 2) not null,              -- 10m高度风速（不可为空，支持两位小数）
                temperature_2m decimal(5, 2) not null,              -- 2m高度温度（不可为空，支持两位小数）
                shortwave_radiation decimal(10, 2),                 -- 向下短波辐照（可为空，支持两位小数）
                unique key (site_id, meteo_id,meteo_times,update_time) -- 联合唯一约束
            );
            """

    # 创建预测结果表
    pred_res_table =  """
            create table if not exists ustlf_pred_res (
            idx bigint auto_increment primary key,  -- 自增主键（BIGINT）
            site_id varchar(50) not null,                -- 电站id（不可为空）
            meteo_id int not null,               -- 气象id（不可为空）
            cal_time datetime not null,          -- 计算时间（不可为空）
            forcast_time_start datetime not null, -- 预测结果的起始时间（不可为空）
            res_data text not null,              -- 预测结果（文本格式，不可为空）
            unique key (site_id, meteo_id, cal_time) -- 联合唯一约束
);
"""

    # 创建日志信息表
    log_info_table = """
    create table if not exists ustlf_log_info (
        log_id bigint auto_increment primary key,       -- 记录id，主键，自增长
        level varchar(20) not null,                     -- 日志级别
        info text not null,                             -- 日志信息
        upload_time datetime not null                   -- 日志录入时间
    );
    """

    # 创建特征超参数信息表
    model_feature_hp_info_table = """
    create table if not exists ustlf_model_feature_hp_info (
        site_id varchar(50) primary key,              -- 电站id，主键
        feature_info text not null,            -- 输入特征信息
        hyperparams_info text not null,         -- 输入超参数信息
        update_time datetime not null           -- 更新时间
    );
    """


    try:
        # 测试数据库连接
        logger.info(f"正在连接数据库: {db.host}:{db.port}")
        db.test_conn()
        logger.info("数据库连接成功")

        # 执行建表语句
        logger.info("正在创建数据表...")

        tables = [
            ('ustlf_station_info', station_info_table),
            ('ustlf_station_history_load', station_history_load_table),
            ('ustlf_meteo_info', meteo_info_table),
            ('ustlf_station_meteo_mapping', station_meteo_mapping_table),
            ('ustlf_station_meteo_data', station_meteo_data_table),
            ('ustlf_station_meteo_forecast_data',station_meteo_forecast_data_table),
            ('ustlf_pred_res', pred_res_table),
            ('ustlf_log_info', log_info_table),
            ('ustlf_model_feature_hp_info', model_feature_hp_info_table)
        ]

        for table_name, create_sql in tables:
            logger.info(f"正在创建{table_name}表...")
            db.execute(create_sql)
            logger.info(f"{table_name}表创建成功")

        logger.info("所有数据表创建成功")

    except Exception as e:
        logger.error(f"创建数据表失败: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        logger.info("开始初始化数据库...")
        config = Config()
        logger.info("配置加载成功")
        logger.info(f"数据库配置: {config.database}")

        db = DataBase(**config.database)
        create_tables(db)

    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        sys.exit(1)
## ustlf超短期负荷预测数据表结构设计



## 1.1 ustlf_station_info 电站信息表

| 列名            | 含义                                    | 备注 |
| --------------- | --------------------------------------- | ---- |
| Site_Id         | 电站id                                  | 主键 |
| Site_Name       | 电站名称                                |      |
| Longitude       | 经度                                    |      |
| Latitude        | 维度                                    |      |
| Stype           | 1：光储电站   2：储能电站   3：其他电站 |      |
| Rated_Capacity  | 额定容量                                |      |
| Rated_Power     | 额定功率                                |      |
| Rated_Power_PV  | 额定光伏发电功率                        |      |
| Frequency_Load  | 负荷数据时间分辨率                      |      |
| Frequency_Meteo | 气象数据分辨率                          |      |
| First_Load_Time | 负荷开始时间                            |      |
| Upload_time     | 电站注册时间                            |      |

 

```sql
CREATE TABLE ustlf_station_info (
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
    First_Load_Time DATETIME,            -- 负荷开始时间（可为空）
    Upload_Time DATETIME NOT NULL        -- 电站注册时间（不可为空）
);

```

## 1.2 ustlf_station_history_load 电站历史负荷表

| 列名           | 含义     | 备注 |
| -------------- | -------- | ---- |
| Site_Id        | 电站id   | 主键 |
| Site_Name      | 电站名称 |      |
| Load_TimeStamp | 负荷时间 |      |
| Load_Data      | 负荷数值 |      |
| Upload_time    | 上传时间 |      |

```sql
CREATE TABLE ustlf_station_history_load (
    Site_Id INT NOT NULL,              -- 电站id（不可为空，外键）
    Site_Name VARCHAR(255) NOT NULL,  -- 电站名称（不可为空）
    Load_TimeStamp DATETIME NOT NULL, -- 负荷时间（不可为空）
    Load_Data DECIMAL(10, 2) NOT NULL, -- 负荷数值（不可为空）
    Upload_Time DATETIME NOT NULL,    -- 上传时间（不可为空）
    PRIMARY KEY (Site_Id, Load_TimeStamp) -- 联合主键：电站id + 负荷时间
);

```



## 1.3 ustlf_meteo_info 气象源信息表

| 列名        | 含义           | 备注 |
| ----------- | -------------- | ---- |
| Meteo_Id    | 气象id         | 主键 |
| Meteo_Name  | 气象名称       |      |
| Upload_Time | 气象源注册时间 |      |

```sql
CREATE TABLE ustlf_meteo_info (
    Meteo_Id INT PRIMARY KEY,         -- 气象id（主键）
    Meteo_Name VARCHAR(255) NOT NULL, -- 气象名称（不可为空）
    Upload_Time DATETIME NOT NULL     -- 气象源注册时间（不可为空）
);

```



## 1.4 ustlf_station_meteo_mapping 电站气象关联表

| 列名        | 含义     | 备注     |
| ----------- | -------- | -------- |
| Site_Id     | 电站id   | 联合主键 |
| Meteo_Id    | 气象id   | 联合主键 |
| Update_Time | 更新时间 |          |

**注**：这里将Site_Id、Meteo_Id作为联合主键

```sql
CREATE TABLE ustlf_station_meteo_mapping (
    Site_Id INT NOT NULL,            -- 电站id（不可为空，联合主键）
    Meteo_Id INT NOT NULL,           -- 气象id（不可为空，联合主键）
    Update_Time DATETIME NOT NULL,   -- 更新时间（不可为空）
    PRIMARY KEY (Site_Id, Meteo_Id)  -- 联合主键：电站id和气象id
);
```



## 1.5 ustlf_station_meteo 电站气象数据表

| 列名                 | 含义         | 备注     |
| -------------------- | ------------ | -------- |
| Site_Id              | 电站id       | 联合主键 |
| Meteo_Id             | 气象id       | 联合主键 |
| Update_Time          | 更新时间     |          |
| Timestamp            | 时间戳       | 联合主键 |
| relative_humidity_2m | 2m相对湿度   |          |
| surface_pressure     | 气压         |          |
| precipitation        | 降水量       |          |
| wind_speed_10m       | 10m高度风速  |          |
| temperature_2m       | 2m高度温度   |          |
| shortwave_radiation  | 向下短波辐照 |          |

```sql
CREATE TABLE ustlf_station_meteo_data (
    Site_Id INT NOT NULL,                     -- 电站id（不可为空，联合主键）
    Meteo_Id INT NOT NULL,                    -- 气象id（不可为空，联合主键）
    Meteo_times DATETIME NOT NULL,            -- 时间戳（不可为空，联合主键）
    Update_Time DATETIME NOT NULL,            -- 更新时间（不可为空）
    relative_humidity_2m DECIMAL(5, 2),       -- 2m相对湿度（可为空，支持两位小数）
    surface_pressure DECIMAL(10, 2),          -- 气压（可为空，支持两位小数）
    precipitation DECIMAL(10, 2),             -- 降水量（可为空，支持两位小数）
    wind_speed_10m DECIMAL(5, 2),             -- 10m高度风速（可为空，支持两位小数）
    temperature_2m DECIMAL(5, 2),             -- 2m高度温度（可为空，支持两位小数）
    shortwave_radiation DECIMAL(10, 2),       -- 向下短波辐照（可为空，支持两位小数）
    PRIMARY KEY (Site_Id, Meteo_Id, Timestamp) -- 联合主键：电站id + 气象id + 时间戳
);

```

## 1.6 ustlf_pred_res 电站预测结果表

| 列名               | 含义               | 备注                                 |
| ------------------ | ------------------ | ------------------------------------ |
| Site_Id            | 电站id             | 联合主键                             |
| Meteo_Id           | 气象id             | 联合主键                             |
| Cal_Time           | 计算时间           | 联合主键                             |
| Forcast_Time_Start | 预测结果的起始时间 | 联合主键                             |
| Res_Data           | 超短期负荷预测结果 | 从Forcast_Time_Start开始的4H预测结果 |

```sql
CREATE TABLE ustlf_pred_res (
    Site_Id INT NOT NULL,              -- 电站id
    Meteo_Id INT NOT NULL,             -- 气象id
    Cal_Time DATETIME NOT NULL,        -- 计算时间
    Forcast_Time_Start DATETIME NOT NULL, -- 预测结果的起始时间
    Res_Data DECIMAL(10, 2),           -- 超短期负荷预测结果
    PRIMARY KEY (Site_Id, Meteo_Id, Cal_Time, Forcast_Time_Start) -- 联合主键
);

```

## 1.7 ustlf_log_info 超短期负荷预测日志记录表

| 列名    | 含义         | 备注 |
| ------- | ------------ | ---- |
| Log_id  | 记录id       | 主键 |
| Site_Id | 电站id       |      |
| Info    | 日志信息     |      |
| Date    | 日志录入时间 |      |

```sql
CREATE TABLE ustlf_log_info (
    Log_id INT AUTO_INCREMENT PRIMARY KEY,  -- 记录id，主键，自增长
    Site_Id INT NOT NULL,                   -- 电站id
    Info TEXT NOT NULL,                     -- 日志信息
    Date DATETIME NOT NULL                  -- 日志录入时间
);

```



## 1.8 ustlf_model_feature_hp_info 输入特征及模型超参数表

| 列名             | 含义           | 备注 |
| ---------------- | -------------- | ---- |
| Site_Id          | 电站id         | 主键 |
| Feature_info     | 输入特征信息   |      |
| Hyperparams_info | 输入超参数信息 |      |

```sql
CREATE TABLE ustlf_model_feature_hp_info (
    Site_Id INT PRIMARY KEY,              -- 电站id，主键
    Feature_info TEXT NOT NULL,            -- 输入特征信息
    Hyperparams_info TEXT NOT NULL         -- 输入超参数信息
);

```


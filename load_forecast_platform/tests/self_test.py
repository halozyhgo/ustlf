# -*- coding: utf-8 -*-
import os
import pytest
import yaml
import pandas as pd
from load_forecast_platform.utils.config import Config
from load_forecast_platform.utils.database import DataBase
# 设置最大显示行数为 100
pd.set_option('display.max_rows', 100)

# 设置最大显示列数为 20
pd.set_option('display.max_columns', 20)
config = Config()
# print(config.config['model_params']['lightgbm'])
#
#
# db = DataBase(**config.database)
# info_query = '''
#             SELECT *
#             FROM ustlf_station_forecast_meteo_data
#             WHERE site_id = '1770070031883964416' AND meteo_times > '2025-1-15 20:00:00' AND meteo_times < '2025-1-16 00:00:00'
#             ORDER BY meteo_times, update_time DESC
#             '''
# df_weather = db.query(info_query)
# df_weather.set_index(pd.to_datetime(df_weather['meteo_times']),inplace=True)
# print(df_weather)
#
# df_weather1 = df_weather[~df_weather.index.duplicated(keep='first')]
# print(df_weather1)

print(config.config['base_features'])





import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    def process_load_data(self, df):
        """处理负荷数据"""
        # 1. 时间格式化
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 2. 排序
        df = df.sort_values('timestamp')
        
        # 3. 去重
        df = df.drop_duplicates('timestamp')
        
        # 4. 异常值处理
        df['load'] = self._handle_outliers(df['load'])
        
        # 5. 缺失值处理
        df = self._handle_missing_values(df)
        
        return df
        
    def process_meteo_data(self, df):
        """处理气象数据"""
        # 1. 时间格式化
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 2. 排序
        df = df.sort_values('timestamp')
        
        # 3. 去重
        df = df.drop_duplicates('timestamp')
        
        # 4. 异常值处理
        for col in ['relative_humidity_2m', 'surface_pressure', 'precipitation',
                   'wind_speed_10m', 'temperation_2m', 'shortwave_radiation']:
            if col in df.columns:
                df[col] = self._handle_outliers(df[col])
        
        # 5. 缺失值处理
        df = self._handle_missing_values(df)
        
        return df
    
    def _handle_outliers(self, series):
        """处理异常值"""
        # 使用3σ原则处理异常值
        mean = series.mean()
        std = series.std()
        series[series > mean + 3*std] = mean + 3*std
        series[series < mean - 3*std] = mean - 3*std
        return series
    
    def _handle_missing_values(self, df):
        """处理缺失值"""
        # 使用线性插值填充缺失值
        return df.interpolate(method='linear') 
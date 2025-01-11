# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from .data_cleaner import DataCleaner
from loguru import logger

class DataProcessor:
    def __init__(self, threshold_loss=0.5, threshold_abnormal=0.5, threshold_constant=0.3):
        self.cleaner = DataCleaner(
            threshold_loss=threshold_loss,
            threshold_abnormal=threshold_abnormal,
            threshold_constant=threshold_constant
        )
    
    def process_load_data(self, df):
        """处理负荷数据"""
        try:
            # 1. 时间格式化
            if 'load_time' in df.columns:
                df.set_index('load_time', inplace=True)
            df.index = pd.to_datetime(df.index)
            
            # 2. 排序
            df = df.sort_index()
            
            # 3. 去重
            df = df[~df.index.duplicated(keep='first')]
            
            # 4. 数据清洗
            df = self.cleaner.clean_load_data(df, load_col='load_data')
            
            logger.info("负荷数据处理完成")
            return df
            
        except Exception as e:
            logger.error(f"负荷数据处理失败: {str(e)}")
            raise
    
    def process_meteo_data(self, df):
        """处理气象数据"""
        # 1. 时间格式化
        df['meteo_times'] = pd.to_datetime(df['meteo_times'])
        
        # 2. 排序
        df = df.sort_values('meteo_times')
        
        # 3. 去重
        df = df.drop_duplicates('meteo_times')
        
        # 4. 异常值处理
        for col in ['relative_humidity_2m', 'surface_pressure', 'precipitation',
                   'wind_speed_10m', 'temperature_2m', 'shortwave_radiation']:
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
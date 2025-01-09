# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from loguru import logger

class DataCleaner:
    def __init__(self, threshold_loss=0.5, threshold_abnormal=0.5, threshold_constant=0.3):
        """
        初始化数据清洗器
        Args:
            threshold_loss: 数据丢失占比的阈值（默认50%）
            threshold_abnormal: 数据大小异常占比的阈值（默认50%）
            threshold_constant: 数据斜率异常占比的阈值（默认30%）
        """
        self.threshold_loss = threshold_loss
        self.threshold_abnormal = threshold_abnormal
        self.threshold_constant = threshold_constant

    def clean_load_data(self, df, load_col='load'):
        """
        清洗负荷数据
        Args:
            df: 包含负荷数据的DataFrame
            load_col: 负荷数据列名
        Returns:
            清洗后的DataFrame
        """
        df = df.copy()
        
        # Step 1: 检测缺失值
        df = self._handle_missing_data(df, load_col)
        
        # Step 2: 检测正常范围外的值
        df = self._handle_abnormal_values(df, load_col)
        
        # Step 3: 检测负荷曲线斜率
        df = self._handle_constant_slope(df, load_col)
        
        # Step 4: 分位法异常值检测与线性插值
        df = self._handle_outliers_and_interpolate(df, load_col)
        
        return df

    def _handle_missing_data(self, df, load_col):
        """处理缺失值"""
        daily_groups = df.groupby(df.index.date)
        
        for day, group in daily_groups:
            missing_count = group[load_col].isna().sum()
            total_count = len(group)
            
            if missing_count/total_count > self.threshold_loss:
                # 获取前一天和后一天的数据
                prev_day = pd.Timestamp(day) - pd.Timedelta(days=1)
                next_day = pd.Timestamp(day) + pd.Timedelta(days=1)
                
                valid_days = []
                for check_day in [prev_day, next_day]:
                    if check_day.date() in daily_groups.groups:
                        check_data = df[df.index.date == check_day.date()]
                        if check_data[load_col].isna().sum()/len(check_data) <= self.threshold_loss:
                            valid_days.append(check_data)
                
                if valid_days:
                    replacement_data = valid_days[-1].copy()
                    time_diff = pd.Timestamp(day) - pd.Timestamp(replacement_data.index[0].date())
                    replacement_data.index = replacement_data.index + time_diff
                    df.loc[df.index.date == day, load_col] = replacement_data[load_col]
                else:
                    df.loc[df.index.date == day, load_col] = df[load_col].mean()
                
                logger.info(f"日期 {day} 的缺失数据已处理")
        
        return df

    def _handle_abnormal_values(self, df, load_col):
        """处理异常值"""
        mean = df[load_col].mean()
        std = df[load_col].std()
        
        # 使用3σ原则检测异常值
        mask = abs(df[load_col] - mean) > 3 * std
        df.loc[mask, load_col] = np.nan
        
        if mask.sum() > 0:
            logger.info(f"检测到 {mask.sum()} 个异常值并已处理")
        
        return df

    def _handle_constant_slope(self, df, load_col):
        """处理恒定斜率"""
        daily_groups = df.groupby(pd.Grouper(freq='D'))
        days_to_replace = []
        
        for day, group in daily_groups:
            slopes = group[load_col].diff() / pd.Timedelta(minutes=15).total_seconds()
            
            constant_slope = 0
            max_constant_slope = 0
            prev_slope = None
            
            for slope in slopes:
                if prev_slope is not None and abs(slope - prev_slope) < 0.001:
                    constant_slope += 1
                    max_constant_slope = max(max_constant_slope, constant_slope)
                else:
                    constant_slope = 0
                prev_slope = slope
            
            if max_constant_slope >= int(96 * self.threshold_constant):  # 96个点代表一天
                days_to_replace.append(day.date())
        
        for day in days_to_replace:
            month = pd.Timestamp(day).month
            month_data = df[df.index.month == month]
            time_of_day = month_data.groupby(
                [month_data.index.hour, month_data.index.minute]
            )[load_col].mean()
            
            day_mask = df.index.date == day
            for idx in df[day_mask].index:
                df.loc[idx, load_col] = time_of_day[idx.hour, idx.minute]
            
            logger.info(f"日期 {day} 的恒定斜率数据已处理")
        
        return df

    def _handle_outliers_and_interpolate(self, df, load_col):
        """处理离群值并进行插值"""
        daily_groups = df.groupby(df.index.date)
        
        for day, group in daily_groups:
            Q1 = group[load_col].quantile(0.25)
            Q3 = group[load_col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            mask = (group[load_col] < lower_bound) | (group[load_col] > upper_bound)
            df.loc[mask.index[mask], load_col] = np.nan
        
        # 使用线性插值填充NaN值
        df[load_col] = df[load_col].interpolate(method='linear')
        
        return df 
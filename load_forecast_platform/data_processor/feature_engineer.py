# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
from chinese_calendar import is_holiday  # 用于判断节假日

class FeatureEngineer:
    def __init__(self):
        """特征工程类"""
        self.time_features = [
            'hour', 'period', 'holiday', 'day_of_week',
            'day_of_month', 'day_of_year', 'month_of_year'
        ]
        
        self.load_features = [
            'before_value', 'shift_value', '4H_value_mean', '4H_value_max', 
            '4H_value_min', 'value_max', 'value_min', 'value_mean', 
            'value_var', 'value_std', 'diff_1', 'diff_2', 'diff_3', 'diff_4'
        ]
        
        self.weather_features = [
            'relative_humidity_2m', 'surface_pressure', 'precipitation',
            'wind_speed_10m', 'temperation_2m', 'shortwave_radiation'
        ]

    def extract_features(self, load_df, meteo_df, horizon=24):
        """
        提取特征
        Args:
            load_df: 负荷数据DataFrame
            meteo_df: 气象数据DataFrame
            horizon: 预测时长(小时)
        Returns:
            特征DataFrame和特征名列表
        """
        try:
            df = load_df.copy()
            df = df.join(meteo_df)
            
            # 1. 时间特征
            df = self._extract_time_features(df)
            
            # 2. 负荷特征
            df = self._extract_load_features(df, horizon)
            
            # 3. 气象特征
            df = self._extract_weather_features(df)
            
            # 获取所有特征列名
            feature_columns = (
                self.time_features + 
                self._get_load_feature_names(horizon) +
                self._get_weather_feature_names()
            )
            
            logger.info(f"特征工程完成，共生成 {len(feature_columns)} 个特征")
            return df, feature_columns
            
        except Exception as e:
            logger.error(f"特征工程失败: {str(e)}")
            raise

    def _extract_time_features(self, df):
        """提取时间特征"""
        df['hour'] = df.index.hour
        df['period'] = df.index.hour * 4 + df.index.minute // 15
        df['day_of_week'] = df.index.weekday
        df['day_of_month'] = df.index.day
        df['day_of_year'] = df.index.dayofyear
        df['month_of_year'] = df.index.month
        df['holiday'] = df.index.map(lambda x: int(is_holiday(x)))
        return df

    def _extract_load_features(self, df, horizon):
        """提取负荷特征"""
        # 基础特征
        df['before_value'] = df['load'].shift(horizon * 4)  # 前H小时负荷
        df['shift_value'] = df['load'].shift(96)  # 前一天同期负荷
        
        # 4小时滑动窗口特征
        df['4H_value_mean'] = df['before_value'].rolling('4H').mean()
        df['4H_value_max'] = df['before_value'].rolling('4H').max()
        df['4H_value_min'] = df['before_value'].rolling('4H').min()
        
        # 24小时滑动窗口特征
        df['value_max'] = df['shift_value'].rolling(96).max()
        df['value_min'] = df['shift_value'].rolling(96).min()
        df['value_mean'] = df['shift_value'].rolling(96).mean()
        df['value_var'] = df['shift_value'].rolling(96).var()
        df['value_std'] = df['shift_value'].rolling(96).std()
        
        # 差分特征
        for i in range(1, 5):
            df[f'diff_{i}'] = df['shift_value'].diff(i)
        
        # 历史负荷特征
        for i in range(1, 8):  # 前7天
            for j in [94, 95, 96, 97, 98]:  # 前后2个点
                col_name = f'last_{i*j}_load'
                df[col_name] = df['load'].shift(i*j)
                self.load_features.append(col_name)
        
        return df

    def _extract_weather_features(self, df):
        """提取气象特征"""
        # 未来24小时气象预报
        for feature in self.weather_features:
            df[f'future_{feature}'] = df[feature].shift(-96)
        return df

    def _get_load_feature_names(self, horizon):
        """获取负荷特征名列表"""
        base_features = self.load_features.copy()
        
        # 添加历史负荷特征名
        for i in range(1, 8):
            for j in [94, 95, 96, 97, 98]:
                base_features.append(f'last_{i*j}_load')
        
        return base_features

    def _get_weather_feature_names(self):
        """获取气象特征名列表"""
        return [f'future_{feature}' for feature in self.weather_features]

    def search_important_features(self, df, target_col='load', n_features=None):
        """
        特征重要性搜索
        Args:
            df: 包含所有特征的DataFrame
            target_col: 目标列名
            n_features: 返回的重要特征数量
        Returns:
            重要特征列表
        """
        try:
            import lightgbm as lgb
            from sklearn.model_selection import train_test_split
            
            # 准备数据
            feature_cols = [col for col in df.columns if col != target_col]
            X = df[feature_cols]
            y = df[target_col]
            
            # 训练模型
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbose': -1
            }
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=100,
                early_stopping_rounds=10
            )
            
            # 获取特征重要性
            importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importance('gain')
            })
            importance = importance.sort_values('importance', ascending=False)
            
            if n_features:
                important_features = importance['feature'].head(n_features).tolist()
            else:
                important_features = importance['feature'].tolist()
                
            logger.info(f"特征重要性搜索完成，选择了 {len(important_features)} 个特征")
            return important_features
            
        except Exception as e:
            logger.error(f"特征重要性搜索失败: {str(e)}")
            raise 
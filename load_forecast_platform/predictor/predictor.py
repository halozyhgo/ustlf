# -*- coding: utf-8 -*-
from load_forecast_platform.utils.logger import setup_logger
import pandas as pd

class LoadPredictor:
    def __init__(self, model, data_processor, site_id):
        self.model = model
        self.data_processor = data_processor
        self.site_id = site_id
        # 创建带有site_id的日志记录器
        self.logger = setup_logger(__name__, site_id=site_id)
        
    def predict(self, features):
        """进行负荷预测"""
        try:
            # 数据预处理
            processed_features = self.data_processor.process_data(features)
            
            # 模型预测
            predictions = self.model.predict(processed_features)
            
            # 整理预测结果
            results = pd.DataFrame({
                'timestamp': features.index,
                'predicted_load': predictions.flatten()
            })
            
            self.logger.info(f"电站 {self.site_id} 预测完成")
            return results
            
        except Exception as e:
            self.logger.error(f"电站 {self.site_id} 预测失败: {str(e)}")
            raise 
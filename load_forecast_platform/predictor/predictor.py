# -*- coding: utf-8 -*-
from utils.logger import setup_logger
import pandas as pd

logger = setup_logger(__name__)

class LoadPredictor:
    def __init__(self, model, data_processor):
        self.model = model
        self.data_processor = data_processor
        
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
            
            logger.info("预测完成")
            return results
            
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            raise 
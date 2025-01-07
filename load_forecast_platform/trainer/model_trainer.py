# -*- coding: utf-8 -*-
from load_forecast_platform.utils.logger import setup_logger
import numpy as np

logger = setup_logger(__name__)

class ModelTrainer:
    def __init__(self, model, training_params):
        self.model = model
        self.training_params = training_params
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        logger.info("开始训练模型...")
        
        try:
            history = self.model.fit(
                X_train, 
                y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                **self.training_params
            )
            
            logger.info("模型训练完成")
            return history
            
        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}")
            raise 
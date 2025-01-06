# -*- coding: utf-8 -*-
import lightgbm as lgb
import numpy as np
from .base_model import BaseModel
import joblib

class LightGBMModel(BaseModel):
    def __init__(self, model_params):
        super().__init__(model_params)
        self.model = None
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val)
            valid_sets = [train_data, valid_data]
            valid_names = ['train', 'valid']
        else:
            valid_sets = [train_data]
            valid_names = ['train']
            
        self.model = lgb.train(
            self.model_params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names
        )
        
        return self.model
        
    def predict(self, X):
        """预测"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
        
    def save(self, path):
        """保存模型"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        joblib.dump(self.model, path)
        
    def load(self, path):
        """加载模型"""
        self.model = joblib.load(path) 
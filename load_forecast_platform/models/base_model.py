# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_params):
        self.model_params = model_params
        self.model = None
        
    @abstractmethod
    def fit(self, X_train, y_train):
        """训练模型"""
        pass
        
    @abstractmethod
    def predict(self, X):
        """预测"""
        pass
        
    @abstractmethod
    def save(self, path):
        """保存模型"""
        pass
        
    @abstractmethod
    def load(self, path):
        """加载模型"""
        pass 
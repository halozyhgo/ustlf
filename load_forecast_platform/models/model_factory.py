# -*- coding: utf-8 -*-
from .base_model import BaseModel
from .lightgbm_model import LightGBMModel

class ModelFactory:
    @staticmethod
    def create_model(model_type: str, model_params: dict) -> BaseModel:
        """创建模型实例"""
        if model_type == 'lightgbm':
            return LightGBMModel(model_params['lightgbm'])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
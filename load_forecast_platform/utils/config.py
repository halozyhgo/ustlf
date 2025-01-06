# -*- coding: utf-8 -*-
import yaml
import os

class Config:
    def __init__(self, config_path=None):
        if config_path is None:
            # 获取默认配置文件路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(os.path.dirname(current_dir), 'configs', 'config.yaml')
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
    @property
    def database(self):
        return self.config['database']
        
    @property
    def model_params(self):
        return self.config['model_params']
        
    @property
    def training_params(self):
        return self.config['training_params'] 
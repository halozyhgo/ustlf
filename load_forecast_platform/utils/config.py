# -*- coding: utf-8 -*-
import yaml
# -*- coding: utf-8 -*-
import os
import yaml
from typing import Dict, Any


class Config:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'configs',
                'config.yaml'
            )

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    @property
    def database(self) -> Dict[str, Any]:
        """获取数据库配置"""
        return {
            'host': self.config['database']['host'],
            'port': self.config['database']['port'],
            'user': self.config['database']['user'],
            'password': self.config['database']['password'],
            'database': self.config['database']['database']
        }
if __name__ == '__main__':
    config = Config('../configs/config.yaml')
    print(config.database)

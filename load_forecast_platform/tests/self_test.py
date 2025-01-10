# -*- coding: utf-8 -*-
import os
import pytest
import yaml
from load_forecast_platform.utils.config import Config

config = Config()
print(config.config['model_params']['lightgbm'])




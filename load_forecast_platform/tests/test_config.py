# -*- coding: utf-8 -*-
import os
import pytest
import yaml
from load_forecast_platform.utils.config import Config

@pytest.fixture
def sample_config_path(tmp_path):
    """创建一个临时的配置文件用于测试"""
    config_data = {
        'database': {
            'url': 'mysql+pymysql://test:test@localhost:3306/test_db',
            'host': 'localhost',
            'port': 3306,
            'user': 'test',
            'password': 'test',
            'database': 'test_db'
        },
        'model_params': {
            'lightgbm': {
                'boosting_type': 'gbdt',
                'objective': 'rmse',
                'n_jobs': 4,
                'max_depth': 8
            }
        },
        'training_params': {
            'batch_size': 32,
            'epochs': 100
        }
    }
    
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f)
    
    return str(config_file)

def test_model_params():
    try:
        config = Config()
        print(config.config['model_params']['lightgbm'])
    except Exception as e:
        pytest.fail(f"加载默认配置文件失败: {str(e)}")

def test_config_load_default():
    """测试默认配置文件加载"""
    try:
        config = Config()
        assert config.config is not None
        assert 'database' in config.config
        assert 'model_params' in config.config
        assert 'training_params' in config.config
    except Exception as e:
        pytest.fail(f"加载默认配置文件失败: {str(e)}")

def test_config_load_custom(sample_config_path):
    """测试自定义配置文件加载"""
    config = Config(config_path=sample_config_path)
    assert config.config is not None
    assert config.database['url'] == 'mysql+pymysql://test:test@localhost:3306/test_db'
    assert config.model_params['lightgbm']['boosting_type'] == 'gbdt'
    assert config.training_params['batch_size'] == 32

def test_database_property(sample_config_path):
    """测试database属性"""
    config = Config(config_path=sample_config_path)
    db_config = config.database
    assert isinstance(db_config, dict)
    assert db_config['host'] == 'localhost'
    assert db_config['port'] == 3306
    assert db_config['database'] == 'test_db'

def test_model_params_property(sample_config_path):
    """测试model_params属性"""
    config = Config(config_path=sample_config_path)
    model_params = config.model_params
    assert isinstance(model_params, dict)
    assert 'lightgbm' in model_params
    assert model_params['lightgbm']['n_jobs'] == 4

def test_training_params_property(sample_config_path):
    """测试training_params属性"""
    config = Config(config_path=sample_config_path)
    training_params = config.training_params
    assert isinstance(training_params, dict)
    assert training_params['epochs'] == 100

def test_config_file_not_found():
    """测试配置文件不存在的情况"""
    with pytest.raises(FileNotFoundError):
        Config(config_path='nonexistent_config.yaml')

def test_invalid_yaml_format(tmp_path):
    """测试无效的YAML格式"""
    invalid_config = tmp_path / "invalid_config.yaml"
    with open(invalid_config, 'w') as f:
        f.write("invalid: yaml: format:")
    
    with pytest.raises(yaml.YAMLError):
        Config(config_path=str(invalid_config))

def test_missing_required_fields(tmp_path):
    """测试缺少必需字段的情况"""
    incomplete_config = {
        'database': {
            'host': 'localhost'
        }
    }
    
    config_file = tmp_path / "incomplete_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(incomplete_config, f)
    
    config = Config(config_path=str(config_file))
    with pytest.raises(KeyError):
        _ = config.model_params 
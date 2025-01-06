import os

def create_project_structure():
    # 项目根目录
    root_dir = "load_forecast_platform"
    
    # 创建目录结构
    structure = {
        'data_processor': ['__init__.py', 'data_loader.py', 'data_cleaner.py', 'feature_engineer.py', 'data_transformer.py'],
        'models': ['__init__.py', 'base_model.py', 'lstm_model.py', 'xgboost_model.py', 'model_factory.py'],
        'trainer': ['__init__.py', 'model_trainer.py', 'param_optimizer.py', 'evaluator.py'],
        'predictor': ['__init__.py', 'predictor.py', 'ensemble.py'],
        'utils': ['__init__.py', 'config.py', 'logger.py', 'db_utils.py'],
        'api': ['__init__.py', 'routes.py', 'schemas.py'],
        'tests': ['__init__.py', 'test_data_processor.py', 'test_models.py', 'test_predictor.py'],
        'scripts': ['train.py', 'predict.py', 'evaluate.py'],
        'configs': ['config.yaml'],
        'docs': ['README.md']
    }
    
    # 创建根目录
    os.makedirs(root_dir, exist_ok=True)
    
    # 创建子目录和文件
    for dir_name, files in structure.items():
        dir_path = os.path.join(root_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        
        for file_name in files:
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_name == 'README.md':
                    f.write('# 超短期负荷预测平台\n\n## 项目说明\n')
                else:
                    f.write('# -*- coding: utf-8 -*-\n\n')

if __name__ == '__main__':
    create_project_structure() 
# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
from load_forecast_platform.utils.config import Config
from load_forecast_platform.data_processor.data_loader import DataLoader
from load_forecast_platform.predictor.predictor import LoadPredictor
from load_forecast_platform.models.model_factory import ModelFactory

app = Flask(__name__)
config = Config()

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/load/historical', methods=['GET'])
def get_historical_load():
    """获取历史负荷数据"""
    try:
        start_time = datetime.strptime(request.args.get('start_time'), '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(request.args.get('end_time'), '%Y-%m-%d %H:%M:%S')
        area_code = request.args.get('area_code')
        
        loader = DataLoader(config.database)
        data = loader.load_load_data(start_time, end_time, area_code)
        
        return jsonify({
            "status": "success",
            "data": data.to_dict('records')
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/weather/historical', methods=['GET'])
def get_historical_weather():
    """获取历史气象数据"""
    try:
        start_time = datetime.strptime(request.args.get('start_time'), '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(request.args.get('end_time'), '%Y-%m-%d %H:%M:%S')
        area_code = request.args.get('area_code')
        
        loader = DataLoader(config.database)
        data = loader.load_weather_data(start_time, end_time, area_code)
        
        return jsonify({
            "status": "success",
            "data": data.to_dict('records')
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/model/train', methods=['POST'])
def train_model():
    """训练模型"""
    try:
        data = request.get_json()
        start_time = datetime.strptime(data['start_time'], '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(data['end_time'], '%Y-%m-%d %H:%M:%S')
        model_type = data['model_type']
        area_code = data.get('area_code')
        
        # 加载数据
        loader = DataLoader(config.database)
        train_data = loader.load_load_data(start_time, end_time, area_code)
        
        # 创建并训练模型
        model = ModelFactory.create_model(model_type, config.model_params)
        trainer = ModelTrainer(model, config.training_params)
        history = trainer.train(train_data)
        
        return jsonify({
            "status": "success",
            "training_history": history
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/predict/load', methods=['POST'])
def predict_load():
    """负荷预测"""
    try:
        data = request.get_json()
        prediction_time = datetime.strptime(data['prediction_time'], '%Y-%m-%d %H:%M:%S')
        model_version = data['model_version']
        horizon = int(data['horizon'])
        area_code = data.get('area_code')
        
        # 加载模型
        model = ModelFactory.load_model(model_version)
        
        # 准备数据
        loader = DataLoader(config.database)
        features = loader.load_features_for_prediction(
            prediction_time,
            horizon,
            area_code
        )
        
        # 预测
        predictor = LoadPredictor(model)
        predictions = predictor.predict(features)
        
        return jsonify({
            "status": "success",
            "predictions": predictions.to_dict('records')
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/model/evaluate', methods=['GET'])
def evaluate_model():
    """评估模型性能"""
    try:
        model_version = request.args.get('model_version')
        start_time = datetime.strptime(request.args.get('start_time'), '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(request.args.get('end_time'), '%Y-%m-%d %H:%M:%S')
        area_code = request.args.get('area_code')
        
        # 加载模型和数据
        model = ModelFactory.load_model(model_version)
        loader = DataLoader(config.database)
        
        # 获取实际值和预测值
        actual_data = loader.load_load_data(start_time, end_time, area_code)
        predictions = model.predict(actual_data)
        
        # 计算评估指标
        evaluation = calculate_metrics(actual_data, predictions)
        
        return jsonify({
            "status": "success",
            "evaluation": evaluation
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500 
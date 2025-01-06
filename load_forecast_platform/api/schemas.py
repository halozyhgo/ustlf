# -*- coding: utf-8 -*-
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict

class LoadData(BaseModel):
    timestamp: datetime
    load_value: float
    area_code: Optional[str]

class WeatherData(BaseModel):
    timestamp: datetime
    temperature: Optional[float]
    humidity: Optional[float]
    wind_speed: Optional[float]
    wind_direction: Optional[float]
    rainfall: Optional[float]
    area_code: Optional[str]

class TrainingParams(BaseModel):
    model_type: str
    start_time: datetime
    end_time: datetime
    area_code: Optional[str]
    custom_params: Optional[Dict]

class PredictionParams(BaseModel):
    model_version: str
    prediction_time: datetime
    horizon: int  # 预测时长(小时)
    area_code: Optional[str]

class PredictionResult(BaseModel):
    timestamp: datetime
    predicted_load: float
    confidence_lower: Optional[float]
    confidence_upper: Optional[float]

class ModelEvaluation(BaseModel):
    model_version: str
    mape: float
    rmse: float
    mae: float
    r2: float
    evaluation_time: datetime 
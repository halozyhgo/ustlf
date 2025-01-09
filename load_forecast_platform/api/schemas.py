# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime, time

class StationRegisterRequest(BaseModel):
    """电站注册请求模型"""
    site_id: str = Field(..., description="电站id")
    site_name: str = Field(..., description="电站名称")
    longitude: float = Field(..., ge=-180, le=180, description="经度")
    latitude: float = Field(..., ge=-90, le=90, description="纬度")
    stype: int = Field(..., description="电站类型")
    rated_capacity: float = Field(..., description="额定容量")
    rated_power: float = Field(..., description="额定功率")
    rated_power_pv: Optional[float] = Field(None, description="额定光伏发电功率")
    frequency_load: int = Field(..., description="负荷数据时间分辨率")
    frequency_meteo: int = Field(..., description="气象数据分辨率")
    # first_load_time: str = Field(..., description="负荷开始时间")
    his_load: list[Any] = Field(..., description="历史负荷数据")
    his_meteo: list[Any] = Field(..., description="历史气象数据")

    @field_validator('longitude')
    @classmethod
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('经度必须在0-180之间')
        return v

    @field_validator('latitude')
    @classmethod
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('纬度必须在-90到90之间')
        return v

class RealTimeDataUploadRequest(BaseModel):
    """实时数据上传请求模型"""
    site_id: str = Field(..., description="电站id")
    real_his_load: Dict[str, Any] = Field(..., description="实时负荷数据")
    real_his_meteo: Dict[str, Any] = Field(..., description="实时气象数据")

    @field_validator('real_his_load')
    @classmethod
    def validate_load_data(cls, v):
        if not isinstance(v, dict) or not v:
            raise ValueError('实时负荷数据不能为空')
        # 检查数据格式
        required_fields = {'timestamp', 'load'}
        if not all(field in v for field in required_fields):
            raise ValueError('负荷数据必须包含 timestamp 和 load 字段')
        return v

    @field_validator('real_his_meteo')
    @classmethod
    def validate_meteo_data(cls, v):
        if not isinstance(v, dict) or not v:
            raise ValueError('实时气象数据不能为空')
        # 检查必需的气象要素
        required_fields = {
            'timestamp', 'relative_humidity_2m', 'surface_pressure',
            'precipitation', 'wind_speed_10m', 'temperation_2m'
        }
        if not all(field in v for field in required_fields):
            raise ValueError(f'气象数据缺少必需字段: {required_fields}')
        return v

class ForecastRequest(BaseModel):
    """预测结果请求模型"""
    site_id: str = Field(..., description="电站id")
    begin_time: str = Field(..., description="预测开始时间")

    @field_validator('begin_time')
    @classmethod
    def validate_begin_time(cls, v):
        try:
            # 验证时间格式 HH:MM
            t = datetime.strptime(v, "%H:%M").time()
            # 检查是否为15分钟的整数倍
            if t.minute % 15 != 0:
                raise ValueError('开始时间必须是15分钟的整数倍')
            return v
        except ValueError as e:
            raise ValueError(f'时间格式错误，应为HH:MM格式: {str(e)}')

class FeatureSearchRequest(BaseModel):
    """输入特征搜索请求模型"""
    site_id: str = Field(..., description="电站id")
    end_date: Optional[str] = Field(None, description="搜索结束时间(默认为最新负荷数据的前一天)")

    @field_validator('end_date')
    @classmethod
    def validate_end_date(cls, v):
        if v is not None:
            try:
                datetime.strptime(v, "%Y-%m-%d")
                return v
            except ValueError:
                raise ValueError('日期格式错误，应为YYYY-MM-DD格式')
        return v

class HyperParamSearchRequest(BaseModel):
    """超参数搜索请求模型"""
    site_id: str = Field(..., description="电站id")
    end_date: Optional[str] = Field(None, description="搜索结束时间(默认为最新负荷数据的前一天)")
    param_ranges: Optional[Dict[str, List[Any]]] = Field(None, description="超参数搜索范围")

    @field_validator('end_date')
    @classmethod
    def validate_end_date(cls, v):
        if v is not None:
            try:
                datetime.strptime(v, "%Y-%m-%d")
                return v
            except ValueError:
                raise ValueError('日期格式错误，应为YYYY-MM-DD格式')
        return v

class HistoryMeteoRequest(BaseModel):
    """历史气象拉取请求模型"""
    site_id: str = Field(..., description="电站id")
    meteo_id: Optional[int] = Field(None, description="气象id(默认为研究院气象)")
    start_time: Optional[str] = Field(None, description="默认为日前3天")
    end_time: Optional[str] = Field(None, description="搜索结束时间(默认为当日19点)")

    @field_validator('end_time')
    @classmethod
    def validate_end_time(cls, v):
        if v is not None:
            try:
                datetime.strptime(v, "%Y-%m-%d")
                return v
            except ValueError:
                raise ValueError('时间格式错误，应为YYYY-MM-DD格式')
        return v

    @field_validator('meteo_id')
    @classmethod
    def validate_meteo_id(cls, v):
        if v is not None and v <= 0:
            raise ValueError('气象id必须大于0')
        return v

class ModelTrainRequest(BaseModel):
    """模型训练请求模型"""
    site_id: str = Field(..., description="电站id")
    end_date: Optional[str] = Field(None, description="输入训练结束时间(默认为当日20点)")
    model_params: Optional[Dict[str, Any]] = Field(None, description="模型参数")
    training_params: Optional[Dict[str, Any]] = Field(None, description="训练参数")

    @field_validator('end_date')
    @classmethod
    def validate_end_date(cls, v):
        if v is not None:
            try:
                datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
                return v
            except ValueError:
                raise ValueError('时间格式错误，应为YYYY-MM-DD HH:MM:SS格式')
        return v

    @field_validator('model_params')
    @classmethod
    def validate_model_params(cls, v):
        if v is not None:
            required_params = {'learning_rate', 'max_depth', 'n_estimators'}
            if not all(param in v for param in required_params):
                raise ValueError(f'模型参数缺少必需字段: {required_params}')
        return v

# 通用响应模型
class CommonResponse(BaseModel):
    """通用响应模型"""
    code: str = Field(..., description="状态码")
    msg: str = Field(..., description="返回信息")
    data: Optional[Dict[str, Any]] = Field(None, description="返回数据")

    @field_validator('code')
    @classmethod
    def validate_code(cls, v):
        valid_codes = {'200', '401', '402', '500'}
        if v not in valid_codes:
            raise ValueError(f'无效的状态码: {v}')
        return v
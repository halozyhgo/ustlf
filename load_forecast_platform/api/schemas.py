# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from datetime import datetime

class StationRegisterRequest(BaseModel):
    """电站注册请求模型"""
    site_id: str = Field(..., description="电站id")
    site_name: str = Field(..., description="电站名称")
    longitude: float = Field(..., ge=0, le=180, description="经度")
    latitude: float = Field(..., ge=0, le=90, description="纬度")
    rated_capacity: float = Field(..., description="额定容量")
    rated_power: float = Field(..., description="额定功率")
    rated_power_pv: Optional[float] = Field(None, description="额定光伏发电功率")
    frequency_load: int = Field(..., description="负荷数据时间分辨率")
    frequency_meteo: int = Field(..., description="气象数据分辨率")
    first_load_time: datetime = Field(..., description="负荷开始时间")
    his_load: Dict[str, Any] = Field(..., description="历史负荷数据")
    his_meteo: Dict[str, Any] = Field(..., description="历史气象数据")

    @validator('longitude')
    def validate_longitude(cls, v):
        if not 0 <= v <= 180:
            raise ValueError('经度必须在0-180之间')
        return v

    @validator('latitude')
    def validate_latitude(cls, v):
        if not 0 <= v <= 90:
            raise ValueError('纬度必须在0-90之间')
        return v

class RealTimeDataUploadRequest(BaseModel):
    """实时数据上传请求模型"""
    site_id: str = Field(..., description="电站id")
    real_his_load: Dict[str, Any] = Field(..., description="实时负荷数据")
    real_his_meteo: Dict[str, Any] = Field(..., description="实时气象数据")

class ForecastRequest(BaseModel):
    """预测结果请求模型"""
    site_id: str = Field(..., description="电站id")
    begin_time: str = Field(..., description="预测开始时间")

class FeatureSearchRequest(BaseModel):
    """输入特征搜索请求模型"""
    site_id: str = Field(..., description="电站id")
    end_date: Optional[str] = Field(None, description="搜索结束时间(默认为最新负荷数据的前一天)")

class HyperParamSearchRequest(BaseModel):
    """超参数搜索请求模型"""
    site_id: str = Field(..., description="电站id")
    end_date: Optional[str] = Field(None, description="搜索结束时间(默认为最新负荷数据的前一天)")

class HistoryMeteoRequest(BaseModel):
    """历史气象拉取请求模型"""
    site_id: str = Field(..., description="电站id")
    meteo_id: Optional[int] = Field(None, description="气象id(默认为研究院气象)")
    end_time: Optional[str] = Field(None, description="搜索结束时间(默认为当日19点)")

class ModelTrainRequest(BaseModel):
    """模型训练请求模型"""
    site_id: str = Field(..., description="电站id")
    end_date: Optional[str] = Field(None, description="输入训练结束时间(默认为当日20点)")

# 通用响应模型
class CommonResponse(BaseModel):
    """通用响应模型"""
    code: str = Field(..., description="状态码")
    msg: str = Field(..., description="返回信息")
    data: Optional[Dict[str, Any]] = Field(None, description="返回数据")

    @validator('code')
    def validate_code(cls, v):
        valid_codes = {'200', '401', '402', '500'}
        if v not in valid_codes:
            raise ValueError(f'无效的状态码: {v}')
        return v

# ... 其他请求模型待实现 
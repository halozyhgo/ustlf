# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class StationRegisterRequest(BaseModel):
    Site_Id: int = Field(..., description="电站id")
    Site_Name: str = Field(..., description="电站名称")
    Longitude: float = Field(..., ge=0, le=180, description="经度")
    Latitude: float = Field(..., ge=0, le=90, description="纬度")
    Stype: int = Field(..., ge=1, le=3, description="电站类型")
    Rated_Capacity: Optional[float] = Field(None, description="额定容量")
    Rated_Power: Optional[float] = Field(None, description="额定功率")
    Rated_Power_PV: Optional[float] = Field(None, description="额定光伏发电功率")
    Frequency_Load: Optional[int] = Field(None, description="负荷数据时间分辨率")
    Frequency_Meteo: Optional[int] = Field(None, description="气象数据分辨率")
    First_Load_Time: Optional[datetime] = Field(None, description="负荷开始时间") 
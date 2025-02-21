# -*- coding: utf-8 -*-
import pandas as pd
from utils.database import DatabaseConnection

class DataLoader:
    def __init__(self, db_config):
        self.db = DatabaseConnection(db_config)
        
    def load_load_data(self, start_time, end_time, area_code=None):
        """加载负荷数据"""
        query = """
            SELECT SiteId, SiteName, LoadTimeStamp, LoadData, UploadTime
            FROM stationhistory_load
            WHERE LoadTimeStamp BETWEEN :start_time AND :end_time
        """
        params = {
            'start_time': start_time,
            'end_time': end_time
        }
        
        return self.db.execute_query(query, params)
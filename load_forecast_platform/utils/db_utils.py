# -*- coding: utf-8 -*-
from sqlalchemy import create_engine
import pandas as pd

class DatabaseConnection:
    def __init__(self, db_config):
        self.db_config = db_config
        self._engine = None
        self._connection = None
        
    @property
    def engine(self):
        if self._engine is None:
            self._engine = create_engine(self.db_config['url'])
        return self._engine
        
    @property
    def connection(self):
        if self._connection is None:
            self._connection = self.engine.connect()
        return self._connection
        
    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None 
# -*- coding: utf-8 -*-
import traceback
import pymysql
import pandas as pd
from loguru import logger

class DataBase(object):
    def __init__(self, host, port, user, password, database):
        """配置数据库信息"""
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self._conn, self._cursor = None, None

    def create_conn(self):
        """创建数据库连接"""
        self._conn = pymysql.connect(
            host=self.host,
            database=self.database,
            port=self.port,
            user=self.user,
            password=self.password,
            charset='utf8',
        )
        self._cursor = self._conn.cursor(cursor=pymysql.cursors.DictCursor)

    def close_conn(self):
        """关闭数据库连接"""
        self._conn.close()
        self._cursor.close()

    def test_conn(self):
        """测试数据库连接"""
        try:
            self.create_conn()
            self.close_conn()
            logger.info('数据库连接测试成功')
        except:
            logger.warning('数据库连接测试失败')

    def execute(self, sql):
        """执行SQL命令"""
        res = None
        try:
            self.create_conn()
            self._cursor.execute(sql)
            self._conn.commit()
            res = self._cursor.fetchall()
            self.close_conn()
        except:
            logger.warning(traceback.print_exc())
            if self._conn.open:
                self._conn.rollback()
                self.close_conn()
        return res

    def query(self, sql):
        """查询并返回DataFrame"""
        df = pd.DataFrame(self.execute(sql))
        return df

    def insert(self, table, df):
        """将DataFrame插入数据库"""
        df = df.astype(object).where(df.notnull(), None)
        df_cols, df_values = df.columns.tolist(), df.values.tolist()

        key, placeholder = '', ''
        for i, col in enumerate(df_cols):
            key += f'`{col}`' if i == (len(df_cols) - 1) else f'`{col}`, '
            placeholder += '%s' if i == (len(df_cols) - 1) else f'%s, '

        res = None
        try:
            self.create_conn()
            self._cursor.executemany(
                f"REPLACE INTO {table}({key}) VALUES({placeholder})", 
                df_values
            )
            self._conn.commit()
            res = self._cursor.fetchall()
            self.close_conn()
        except:
            logger.warning(traceback.print_exc())
            self._conn.rollback()

        return res

    def delete(self, table, cond):
        """删除满足条件的数据"""
        sql = f"DELETE FROM {table} WHERE {cond}"
        res = self.execute(sql)
        return res 
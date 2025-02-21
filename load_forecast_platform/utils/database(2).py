import traceback
import pymysql
import pandas as pd
from loguru import logger
from contextlib import contextmanager, closing

class DataBase(object):

    def __init__(self, host, port, user, password, database):
        """
        配置数据库信息
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database

    @contextmanager
    def get_cursor(self):
        """
        使用上下文管理器获取数据库连接和游标
        """
        conn = None
        try:
            conn = pymysql.connect(
                host=self.host,
                database=self.database,
                port=self.port,
                user=self.user,
                password=self.password,
                charset='utf8',
            )
            cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
            yield cursor
            conn.commit()
        except Exception as e:
            logger.warning(f"Error during database operation: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                cursor.close()
                conn.close()

    def test_conn(self):
        """
        测试数据库连接是否正常
        """
        try:
            with self.get_cursor() as cursor:
                logger.info('数据库连接测试成功')
        except:
            logger.warning('数据库连接测试失败')

    def execute(self, sql):
        """
        执行数据库命令
        :param sql: str - SQL查询语句
        :return: 返回查询结果或 None
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(sql)
                return cursor.fetchall()
        except Exception as e:
            logger.warning(f"Error executing SQL: {e}")
            return None

    def query(self, sql):
        """
        查询数据表并返回dataframe
        :param sql: str - SQL查询语句
        :return: pd.DataFrame - 查询结果
        """
        res = self.execute(sql)
        return pd.DataFrame(res) if res is not None else pd.DataFrame()

    def insert(self, table, df):
        """
        将dataframe插入数据库，支持单行和多行插入
        :param table: str - 数据库表名
        :param df: pd.DataFrame - 需要插入的数据
        :return: 成功返回 True，失败返回 False
        """
        # 数据预处理
        if df.empty:
            logger.warning(f"No data to insert into table {table}")
            return False
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        df = df.astype(object).where(df.notnull(), None)
        df_cols = df.columns.tolist()
        df_values = df.values.tolist()

        # 格式化SQL语句
        key = ', '.join([f'`{col}`' for col in df_cols])
        placeholder = ', '.join(['%s'] * len(df_cols))
        on_duplicate = ', '.join([f'`{col}`=VALUES(`{col}`)' for col in df_cols])
        sql = f""" INSERT INTO {table} ({key}) VALUES ({placeholder}) ON DUPLICATE KEY UPDATE {on_duplicate}"""

        try:
            with self.get_cursor() as cursor:
                cursor.executemany(sql, df_values)
            # logger.info(f"Successfully inserted {total_rows} rows into {table}")
            return True
        except Exception as e:
            logger.error(f"Error inserting data into {table}: {e}")
            traceback.print_exc()
            return False

    def delete(self, table, cond):
        """
        从数据表中删除满足条件的数据
        :param table: str - 数据库表名
        :param cond: str - 删除条件
        :return: 成功返回 True，失败返回 False
        """
        sql = f"DELETE FROM {table} WHERE {cond}"
        try:
            with self.get_cursor() as cursor:
                cursor.execute(sql)
            logger.info(f"Deleted data from {table} where {cond}")
            return True
        except Exception as e:
            logger.error(f"Error deleting data from {table}: {e}")
            return False

if __name__ == '__main__':

    # 初始化
    DB = DataBase('121.40.64.25', 30306, 'root', 'lqJeaFfVmD', 'vpp')
    DB.test_conn()

    # 创建表, 设置时间唯一
    table = 'test'
    sql = f"""
    CREATE TABLE IF NOT EXISTS `{table}` (
        `id` bigint NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '主键id',             
        `time` datetime NOT NULL UNIQUE COMMENT '时间',
        `value` float COMMENT '预测值'
    )
    """
    # DB.execute(sql)

    # 插入数据
    df = pd.DataFrame(pd.date_range('2024-01-01', periods=24 * 7, freq='1h'), columns=['time'])
    df['value'] = list(range(len(df)))
    DB.insert(table, df)

    # 查询数据
    df = DB.query(f'SELECT * FROM {table}')

    # 更新数据
    df = pd.DataFrame(pd.date_range('2024-01-01', periods=24, freq='1h'), columns=['time'])
    df['value'] = 100
    DB.insert(table, df)

    # 查询数据
    sql = f"SELECT time, value FROM {table} WHERE time >= '2024-01-01' AND time < '2024-01-02'"
    df = DB.query(sql)
    print(df)

    # # 删除表
    # DB.execute(f'DROP TABLE IF EXISTS {table}')
    #
    # # 清空表
    # DB.execute(f'TRUNCATE TABLE {table}')
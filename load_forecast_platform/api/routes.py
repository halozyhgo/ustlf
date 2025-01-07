from flask import request, jsonify
from load_forecast_platform.api import app
from load_forecast_platform.utils.db_utils import DatabaseConnection
from load_forecast_platform.utils.config import Config
from load_forecast_platform.data_processor.data_processor import DataProcessor
import pandas as pd
from datetime import datetime
from sqlalchemy import text

@app.route('/ustlf/station/register', methods=['POST'])
def register_station():
    """电站注册接口"""
    try:
        # 1. 参数校验
        data = request.json
        validation_result = validate_station_params(data)
        if not validation_result['success']:
            return jsonify({
                'code': 401,
                'msg': f"参数校验失败: {validation_result['msg']}"
            })

        # 2. 获取参数
        site_info = {
            'Site_Id': data['Site_Id'],
            'Site_Name': data['Site_Name'],
            'Longitude': data['Longitude'],
            'Latitude': data['Latitude'],
            'Stype': data['Stype'],
            'Rated_Capacity': data.get('Rated_Capacity'),
            'Rated_Power': data.get('Rated_Power'),
            'Rated_Power_PV': data.get('Rated_Power_PV'),
            'Frequency_Load': data.get('Frequency_Load'),
            'Frequency_Meteo': data.get('Frequency_Meteo'),
            'First_Load_Time': data.get('First_Load_Time'),
            'Upload_Time': datetime.now()
        }

        # 3. 处理历史负荷数据
        his_load_file = request.files.get('His_load')
        if his_load_file:
            load_data = pd.read_csv(his_load_file)
            # 数据处理
            processor = DataProcessor()
            processed_load = processor.process_load_data(load_data)

        # 4. 处理历史气象数据
        his_meteo_file = request.files.get('His_meteo')
        if his_meteo_file:
            meteo_data = pd.read_csv(his_meteo_file)
            # 数据处理
            processed_meteo = processor.process_meteo_data(meteo_data)

        # 5. 存储数据
        db = DatabaseConnection(Config().database)
        with db.engine.begin() as conn:
            # 存储电站信息
            insert_station_sql = """
                INSERT INTO ustlf_station_info (
                    Site_Id, Site_Name, Longitude, Latitude, Stype,
                    Rated_Capacity, Rated_Power, Rated_Power_PV,
                    Frequency_Load, Frequency_Meteo, First_Load_Time, Upload_Time
                ) VALUES (
                    :Site_Id, :Site_Name, :Longitude, :Latitude, :Stype,
                    :Rated_Capacity, :Rated_Power, :Rated_Power_PV,
                    :Frequency_Load, :Frequency_Meteo, :First_Load_Time, :Upload_Time
                )
            """
            conn.execute(text(insert_station_sql), site_info)

            # 存储历史负荷数据
            if his_load_file:
                for _, row in processed_load.iterrows():
                    insert_load_sql = """
                        INSERT INTO ustlf_station_history_load (
                            Site_Id, Site_Name, Load_TimeStamp, Load_Data, Upload_Time
                        ) VALUES (
                            :Site_Id, :Site_Name, :Load_TimeStamp, :Load_Data, :Upload_Time
                        )
                    """
                    load_data = {
                        'Site_Id': site_info['Site_Id'],
                        'Site_Name': site_info['Site_Name'],
                        'Load_TimeStamp': row['timestamp'],
                        'Load_Data': row['load'],
                        'Upload_Time': datetime.now()
                    }
                    conn.execute(text(insert_load_sql), load_data)

            # 存储气象数据
            if his_meteo_file:
                for _, row in processed_meteo.iterrows():
                    insert_meteo_sql = """
                        INSERT INTO ustlf_station_meteo_data (
                            Site_Id, Meteo_Id, Meteo_times, Update_Time,
                            relative_humidity_2m, surface_pressure, precipitation,
                            wind_speed_10m, temperation_2m, shortwave_radiation
                        ) VALUES (
                            :Site_Id, :Meteo_Id, :Meteo_times, :Update_Time,
                            :relative_humidity_2m, :surface_pressure, :precipitation,
                            :wind_speed_10m, :temperation_2m, :shortwave_radiation
                        )
                    """
                    meteo_data = {
                        'Site_Id': site_info['Site_Id'],
                        'Meteo_Id': 1,  # 这里需要根据实际情况设置
                        'Meteo_times': row['timestamp'],
                        'Update_Time': datetime.now(),
                        'relative_humidity_2m': row.get('relative_humidity_2m'),
                        'surface_pressure': row.get('surface_pressure'),
                        'precipitation': row.get('precipitation'),
                        'wind_speed_10m': row.get('wind_speed_10m'),
                        'temperation_2m': row.get('temperation_2m'),
                        'shortwave_radiation': row.get('shortwave_radiation')
                    }
                    conn.execute(text(insert_meteo_sql), meteo_data)

        return jsonify({
            'code': 200,
            'msg': '电站注册成功'
        })

    except Exception as e:
        return jsonify({
            'code': 500,
            'msg': f'服务器错误: {str(e)}'
        })


def validate_station_params(data):
    """验证电站注册参数"""
    required_fields = {
        'Site_Id': int,
        'Site_Name': str,
        'Longitude': float,
        'Latitude': float,
        'Stype': int
    }

    # 检查必填字段
    for field, field_type in required_fields.items():
        if field not in data:
            return {'success': False, 'msg': f'缺少必填字段 {field}'}
        if not isinstance(data[field], field_type):
            return {'success': False, 'msg': f'字段 {field} 类型错误'}

    # 检查数值范围
    if not (0 <= data['Longitude'] <= 180):
        return {'success': False, 'msg': '经度范围应在0-180之间'}
    if not (0 <= data['Latitude'] <= 90):
        return {'success': False, 'msg': '纬度范围应在0-90之间'}
    if data['Stype'] not in [1, 2, 3]:
        return {'success': False, 'msg': '电站类型应为1、2或3'}

    return {'success': True, 'msg': '验证通过'}
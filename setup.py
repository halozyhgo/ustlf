from setuptools import setup, find_packages

setup(
    name="load_forecast_platform",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'sqlalchemy',
        'pymysql',
        'pandas',
        'lightgbm',
        'flask'
    ]
) 
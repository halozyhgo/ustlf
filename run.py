# -*- coding: utf-8 -*-
from load_forecast_platform.app import create_app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000) 
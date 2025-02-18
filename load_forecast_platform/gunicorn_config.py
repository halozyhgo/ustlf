# -*- coding: utf-8 -*-
import multiprocessing

# Server socket
# bind = '0.0.0.0:5000'
bind = '192.168.153.227:5555'
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'gevent'
worker_connections = 1000
timeout = 120
keepalive = 2

# Process naming
proc_name = 'load_forecast_platform'

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Maximum number of requests a worker will process before restarting
max_requests = 2000
max_requests_jitter = 400

# Process management
preload_app = True
reload = True 
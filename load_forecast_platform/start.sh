#!/bin/bash
gunicorn -c gunicorn_config.py 'api:app' 
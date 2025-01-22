# -*- coding: utf-8 -*-
from flask import Flask

app = Flask(__name__)

from . import routes
if __name__ == '__main__':
    # app.run(debug=True)
    # app.run(processes=4)
    from gevent import pywsgi
    app.debug=True
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    server.serve_forever()
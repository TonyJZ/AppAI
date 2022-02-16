from flask import Flask, redirect, url_for, request
import time
from datetime import datetime

app = Flask(__name__)


# Main end point for RESTful API
@app.route('/', methods = ['POST'])
def post_recv():
    # Obtain content object
    # print(request)
    content = request.get_json()
    # meter_id = content['meter_id']
    # print(request.is_json)
    # content = request.get_json()
    print(content, datetime.now())
    # print(content['meter_id'], content['solution_type'])
    # time.sleep(1)
    return 'ok'


if __name__ == '__main__':
    app.run(threaded=True, port='5002')

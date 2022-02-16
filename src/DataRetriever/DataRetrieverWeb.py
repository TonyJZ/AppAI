import json
from datetime import datetime

from flask import Flask, request, Response

from lib.Log.Log import Logger
from src.DataRetriever.DataRetrieverRequestHandlerAbc import DataRetrieverRequestHandlerAbc

_PREDICTION_RESULT_QUERY_TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S.%f'


class DataRetrieverWeb:

    def __init__(self, request_handler):
        Logger.debug("Initializing Data Retriever Web Module")

        self.request_handler: DataRetrieverRequestHandlerAbc = request_handler

        self.app = None
        self.create_app()

    def create_app(self):
        Logger.debug('Creating Flask App')
        app = Flask(__name__)

        @app.route('/prediction-results', methods=['POST'])
        def handle_query_prediction_results():

            # convert request body to json
            request_json_body = request.get_json()
            Logger.info('Received request {}'.format(request_json_body))

            # reading the json obj
            type_id = request_json_body.get('type_id')
            pipeline = request_json_body.get('pipeline')

            try:
                start_time = datetime.strptime(request_json_body.get('start_date'),
                                               _PREDICTION_RESULT_QUERY_TIMESTAMP_FORMAT)
            except TypeError as e:
                Logger.error(e)
                start_time = None

            try:
                end_time = datetime.strptime(request_json_body.get('end_date'),
                                             _PREDICTION_RESULT_QUERY_TIMESTAMP_FORMAT)
            except TypeError as e:
                Logger.error(e)
                end_time = None

            # create the Response object
            resp = Response()

            # validate the inputs
            if type_id is None or pipeline is None or start_time is None or end_time is None:
                resp.status_code = 400
                Logger.info('Incorrect body parameters')
            else:
                try:

                    data = self.request_handler.query_prediction_results(pipeline, type_id, start_time,
                                                                         end_time)
                    resp.set_data(json.dumps(data))
                except Exception as e:
                    print(e)
                    Logger.error(e)
                    resp.status_code = 500

            Logger.info('Returning Response {}'.format(resp))
            return resp

        @app.route('/model-training-info', methods=['POST'])
        def handle_query_model_training_info():

            # convert request body to json
            request_json_body = request.get_json()
            Logger.info('Received request {}'.format(request_json_body))

            # reading the json obj
            type_id = request_json_body.get('type_id')
            pipeline = request_json_body.get('pipeline')

            resp = Response()
            if type_id is None or pipeline is None:
                resp.status_code = 400
                Logger.info('Incorrect body parameters')
            else:
                try:
                    # Todo: call method from Data Retriever Request Handler
                    data = self.request_handler.query_all_training_model_info(pipeline, type_id)
                    # print(data)
                    # a = json.dumps(data)
                    # print(json.dumps(data))
                    resp.set_data(json.dumps(data))
                except Exception as e:
                    Logger.error(e)
                    print(e)
                    resp.status_code = 500

            Logger.info('Returning Response {}'.format(resp))
            return resp

        @app.route('/model-training-info/details', methods=['POST'])
        def handle_query_model_training_detail_info():

            # convert request body to json
            request_json_body = request.get_json()
            Logger.info('Received request {}'.format(request_json_body))

            # reading the json obj
            type_id = request_json_body.get('type_id')
            pipeline = request_json_body.get('pipeline')
            try:
                target_date = datetime.strptime(request_json_body.get('target_date'),
                                                _PREDICTION_RESULT_QUERY_TIMESTAMP_FORMAT)
            except (TypeError, ValueError) as e:
                Logger.error(e)
                target_date = None

            resp = Response()
            if type_id is None or pipeline is None or target_date is None:
                resp.status_code = 400
                Logger.info('Incorrect body parameters')
            else:
                try:
                    data = self.request_handler.query_training_model_info(pipeline, type_id, target_date)
                    resp.set_data(json.dumps(data))
                except Exception as e:
                    Logger.error(e)
                    resp.status_code = 500

            Logger.info('Returning Response {}'.format(resp))
            return resp

        self.app = app

    def start(self, **kwargs):
        self.app.run(**kwargs)

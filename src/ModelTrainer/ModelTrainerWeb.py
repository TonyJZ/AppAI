from flask import Flask, request, Response

from lib.Log.Log import Logger


class ModelTrainerWeb:

    def __init__(self):
        Logger.debug("Initializing Model Trainer Web Module")

    def create_app(self, request_processor):
        Logger.debug('Creating Flask App')
        app = Flask(__name__)
        trainer_processor = request_processor

        @app.route('/', methods=['POST'])
        def handle_train_model_request():
            request_json = request.get_json()
            Logger.debug('Received request {}, json_content={}'.format(request, request_json))
            type_id = request_json.get('type_id')
            pipeline_name = request_json.get('pipeline_name')
            stime = request_json.get('s_time')
            etime = request_json.get('e_time')

            resp = Response()
            try:
                # trainer_processor.serialize_training(pipeline, type_id)

                trainer_processor.parallelize_training(pipeline_name, type_id, stime, etime)
                resp.set_data('ok')

            except Exception as e:
                resp.status_code = 500
                Logger.error(e)

            Logger.debug('Returning Response {}'.format(resp))
            return resp

        @app.route('/SubmitTrainingResult', methods=['POST'])
        def handle_train_result():
            request_json = request.get_json()

            resp = Response()
            try:
                trainer_processor.process_training_result(request_json)
                resp.set_data('ok')
            except Exception as e:
                resp.status_code = 500
                Logger.error(e)

            Logger.debug('Returning Response {}'.format(resp))
            return resp

        return app


if __name__ == '__main__':
    print('RUNNING MODEL TRAINER WEB API ONLY')
    ModelTrainerWeb().create_app().run()
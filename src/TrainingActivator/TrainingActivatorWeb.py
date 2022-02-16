from flask import Flask, request, Response

from lib.Exceptions.AsyncWebException import ModelTrainerInternalProcessingException
from lib.Log.Log import Logger
from src.TrainingActivator.TrainingActivatorAsyncWebRequester import TrainingActivatorAsyncWebRequester


class TrainingActivatorWeb():

    def __init__(self, requester, loop):
        Logger.debug("Initializing Training Activator Web Module")
        self.requester: TrainingActivatorAsyncWebRequester = requester
        self.loop = loop

    def create_app(self):
        Logger.debug('Creating Flask App')
        app = Flask(__name__)

        @app.route('/', methods=['POST'])
        def handle_user_training_command():
            request_json_body = request.get_json()
            Logger.debug('Received request {}'.format(request_json_body))
            type_id = request_json_body.get('type_id')
            pipeline = request_json_body.get('pipeline_name')
            stime = request_json_body.get('training_set_start_time')
            etime = request_json_body.get('training_set_end_time')
            resp = Response()
            if type_id is None or pipeline is None:
                resp.status_code = 400
                Logger.info('Incorrect body parameters')
            else:
                try:

                    self.loop.run_until_complete(
                        self.requester.post_to_model_trainer_from_request(type_id=type_id, pipeline=pipeline, stime=stime, etime=etime))

                    resp.set_data('ok')
                except ModelTrainerInternalProcessingException as modelTrainerInternalProcessingException:
                    resp.status_code = 500
                except Exception as e:
                    resp.status_code = 500

            Logger.debug('Returning Response {}'.format(resp))
            return resp

        return app


if __name__ == '__main__':
    print('RUNNING TRAINING ACTIVATOR WEB API ONLY')
    TrainingActivatorWeb().create_app().run()

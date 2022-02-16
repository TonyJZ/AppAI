import sys

from src.Pipelines.PipelineFactory import PipelineFactory
from src.ModelTrainer.ModelTrainerProcessing import ModelTrainerProcessing
from src.ModelTrainer.ModelTrainerConfig import ModelTrainerConfig
from src.ModelTrainer.ModelTrainerWeb import ModelTrainerWeb
from lib.Log.Log import Logger


class ModelTrainer:

    def __init__(self):
        Logger.debug('Initializing the Model Trainer')

        # initialize the trainer process object
        self.trainer_processor = ModelTrainerProcessing()
        # create web app
        self.app = ModelTrainerWeb().create_app(self.trainer_processor)

        Logger.debug('Creating sub-process for handling the remote results')

        # initialize Pipeline factory
        PipelineFactory()

    def run(self):

        try:

            Logger.debug('Starting Web App at port[%d]', ModelTrainerConfig.web_api_port)
            self.app.run(threaded=True, host='0.0.0.0', port=ModelTrainerConfig.web_api_port)
            # self.app.run(threaded=True, port=ModelTrainerConfig.web_api_port)
        except KeyboardInterrupt:
            Logger.info('Program is Interrupted by Keyboard Inputs')

        Logger.info('Program is exiting')
        sys.exit()


if __name__ == '__main__':
    modelTrainer = ModelTrainer()
    modelTrainer.run()

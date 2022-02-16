import asyncio
import sys
import multiprocessing as mp

from src.TrainingActivator.TrainingActivatorAsyncWebRequester import TrainingActivatorAsyncWebRequester
from src.TrainingActivator.TrainingScheduler import training_scheduler_process, TrainingScheduler
from lib.Log.Log import Logger
from src.TrainingActivator.TrainingActivatorConfig import TrainingActivatorConfig
from src.TrainingActivator.TrainingActivatorWeb import TrainingActivatorWeb


class TrainingActivator:

    def __init__(self):
        Logger.debug('Initializing the Training Activator')
        self.processes = []

    def run(self):
        Logger.debug('Creating sub-process for Scheduler')

        # initializing the requester
        requester = TrainingActivatorAsyncWebRequester(TrainingActivatorConfig.model_trainer_url)

        self.processes.append(mp.Process(target=training_scheduler_process, args=[requester]))

        # start all the sub-processes
        Logger.debug('Starting all the sub processes')
        for process in self.processes:
            process.start()

        # setup event loop for main thread
        policy = asyncio.get_event_loop_policy()
        policy.set_event_loop(policy.new_event_loop())
        loop = asyncio.new_event_loop()

        try:
            # get the flask app
            app = TrainingActivatorWeb(requester, loop).create_app()
            # start the flask app
            Logger.debug('Starting Web App at port[%d]', TrainingActivatorConfig.web_api_port)
            app.run(threaded=True, host='0.0.0.0', port=TrainingActivatorConfig.web_api_port)
            # app.run(threaded=True, port=TrainingActivatorConfig.web_api_port)
        except KeyboardInterrupt:
            Logger.info('Program is Interrupted by Keyboard Inputs')

        loop.close()

        Logger.info('Program is exiting')
        sys.exit()


if __name__ == '__main__':
    trainingActivator = TrainingActivator()
    trainingActivator.run()


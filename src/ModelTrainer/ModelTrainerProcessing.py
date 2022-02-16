import time
import dateutil.parser
import ray

from src.ModelTrainer.ModelTrainerAsyncWebRequester import ModelTrainerAsyncWebRequester
from Pipelines.PipelineFactory import PipelineFactory
from lib.Log.Log import Logger
from src.ModelTrainer.ModelTrainerConfig import ModelTrainerConfig


class ModelTrainerProcessing:
    __ray = ray

    def __init__(self):
        ModelTrainerProcessing.__ray.init(num_cpus=ModelTrainerConfig.ray_cpus,
                                          object_store_memory=ModelTrainerConfig.ray_memory)

        self.requester = ModelTrainerAsyncWebRequester(ModelTrainerConfig.internal_training_result_url)

    # def __del__(self):
    #     print('__del__ for ModelTrainerProcessing')
    #     if ModelTrainerProcessing.__ray.is_initialized():
    #         ModelTrainerProcessing.__ray.shutdown()

    def parallelize_training(self, pipeline_name: str, type_id: int, stime=None, etime=None):
        # get pipeline from factory
        pipeline = PipelineFactory().get_pipeline(pipeline_name)
        Logger.debug('Pipeline name [{}], got pipeline {} from PipelineFactory'.format(pipeline_name, pipeline))
        pipeline_config = pipeline.get_config()
        pipeline_config.setdefault('web_requester', self.requester)

        if stime is not None and etime is not None:
            pipeline_config.setdefault('s_time', dateutil.parser.parse(stime))
            pipeline_config.setdefault('e_time', dateutil.parser.parse(etime))

        # pipeline.get_class().train(type_id, pipeline.get_config())

        # wrap the function by ray remote
        remote_func = ModelTrainerProcessing.__ray.remote(pipeline.get_class().train)
        remote_func.remote(type_id, pipeline_config)

    def serialize_training(self, pipeline_name: str, type_id: int, stime=None, etime=None) -> None:
        # get pipeline from factory
        pipeline = PipelineFactory().get_pipeline(pipeline_name)
        Logger.debug('Pipeline name [{}], got pipeline {} from PipelineFactor'.format(pipeline_name, pipeline))

        pipeline_config = pipeline.get_config()
        pipeline_config.setdefault('web_requester', self.requester)

        if stime is not None and etime is not None:
            pipeline_config.setdefault('s_time', dateutil.parser.parse(stime))
            pipeline_config.setdefault('e_time', dateutil.parser.parse(etime))

        pipeline.get_class().train(type_id, pipeline_config)

    def process_training_result(self, result_json):
        Logger.info("Training Result: %s", result_json)


if __name__ == '__main__':
    trainer_processor = ModelTrainerProcessing()
    # trainer_processor.parallelize_training("CNAF1", 3)
    # trainer_processor.parallelize_training("SAGW2", 19)
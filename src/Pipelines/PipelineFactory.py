import importlib
from typing import Dict

from Pipelines.ProcessPipeline import ProcessPipeline
from lib.Config.ConfigUtil import ConfigUtil
from lib.Log.Log import Logger


class PipelineFactory(object):
    __pipelines: Dict[str, ProcessPipeline] = None

    def __init__(self):
        # load configuration and create a list of pipelines
        if PipelineFactory.__pipelines is None:
            Logger.info('initialize the pipelines')
            PipelineFactory.__pipelines = {}
            stream_configs: Dict = ConfigUtil.get_value_by_key('streams')
            for key, value in stream_configs.items():
                stream_name = value.get('stream')
                pipelines = value.get('pipelines')
                for pipeline_config in pipelines:
                    pipeline_config['stream'] = stream_name
                    pipeline: ProcessPipeline = self.create_pipeline(pipeline_config)
                    PipelineFactory.__pipelines.setdefault(pipeline_config.get('pipeline_name'), pipeline)

    def create_pipeline(self, config) -> ProcessPipeline:

        class_name = config.get('pipeline_name') + 'Pipeline'
        module = importlib.import_module('.' + class_name, package='Pipelines')
        pipeline = getattr(module, class_name)(config)

        return pipeline

    def get_pipeline(self, pipeline_name) -> ProcessPipeline:
        return PipelineFactory.__pipelines.get(pipeline_name)

    def get_pipeline_class(self, pipeline_name) -> ProcessPipeline:
        return PipelineFactory.__pipelines.get(pipeline_name).get_class()

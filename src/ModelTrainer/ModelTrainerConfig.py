from typing import Dict

from lib.Config.ConfigUtil import ConfigUtil

_CONFIG_SERVICE_KEY = 'services'
_CONFIG_MODEL_TRAINER_KEY = 'ModelTrainer'


class __ModelTrainerConfig:

    def __init__(self):
        model_trainer_config: Dict[str, any] = ConfigUtil.get_value_by_key(_CONFIG_SERVICE_KEY,
                                                                           _CONFIG_MODEL_TRAINER_KEY)
        self.web_api_port: str = model_trainer_config.get('web_api_port')
        self.ray_cpus: int = model_trainer_config.get('ray_cpus')
        self.ray_memory: int = model_trainer_config.get('ray_memory')

        self.pipelines_dict = {}
        stream_config = ConfigUtil.get_value_by_key('streams')
        for stream in stream_config:
            for pipeline in stream_config[stream]['pipelines']:
                self.pipelines_dict.setdefault(pipeline.get('pipeline_name'), pipeline)

        self.internal_training_result_url = 'http://{}:{}/SubmitTrainingResult'.format(
            model_trainer_config.get('host_ip'), self.web_api_port)


ModelTrainerConfig: __ModelTrainerConfig = __ModelTrainerConfig()

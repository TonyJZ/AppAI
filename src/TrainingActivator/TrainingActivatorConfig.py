from typing import List

from lib.Config.ConfigUtil import ConfigUtil
from lib.Log.Log import Logger

_CONFIG_SERVICE_KEY = 'services'
_CONFIG_TRAINING_ACTIVATOR_KEY = 'TrainingActivator'


class __TrainingActivatorConfig:
    def __init__(self):

        self.web_api_port = ConfigUtil.get_value_by_key(_CONFIG_SERVICE_KEY, _CONFIG_TRAINING_ACTIVATOR_KEY,
                                                        'web_api_port')
        self.model_trainer_url = 'http://' + ConfigUtil.get_value_by_key(_CONFIG_SERVICE_KEY,
                                                                         _CONFIG_TRAINING_ACTIVATOR_KEY,
                                                                         'model_trainer_ip_port')

        self.log_level = ConfigUtil.get_value_by_key(_CONFIG_SERVICE_KEY, _CONFIG_TRAINING_ACTIVATOR_KEY, 'logging',
                                                     'level')
        self.log_path = ConfigUtil.get_value_by_key(_CONFIG_SERVICE_KEY, _CONFIG_TRAINING_ACTIVATOR_KEY, 'logging',
                                                    'path')

        self.pipelines = []
        stream_config = ConfigUtil.get_value_by_key('streams')
        for stream in stream_config:
            for pipeline in stream_config[stream]['pipelines']:
                pipeline_config = dict()
                pipeline_config.setdefault('type_ids', stream_config[stream]['type_ids'])
                pipeline_config.setdefault('pipeline_name', pipeline.get('pipeline_name'))
                # pipeline_config.setdefault('training_interval', pipeline.get('training_interval'))

                training_activation_config = pipeline.get('training_activation')
                hour, minute, second = training_activation_config.get('time').split(':')
                cron_schedule = {
                    'hour': hour,
                    'minute': minute,
                    'second': second,
                }
                day_list: List = training_activation_config.get('day')
                weekday_list: List = training_activation_config.get('weekday')

                if day_list is not None and 1 == len(day_list) and 0 < day_list[0] <= 31:
                    # set to activate once a month
                    cron_schedule.setdefault('day', day_list[0])

                elif type(weekday_list) is list:
                    if 1 == len(weekday_list):
                        if 0 <= weekday_list[0] <= 6:
                            # set to activate only once per week
                            cron_schedule.setdefault('day_of_week', weekday_list[0])

                    elif 7 >= len(weekday_list):
                        # verify the values in the weekday_list
                        prev_index = -1
                        weekday_list.sort()
                        for item in weekday_list:
                            # validate the range of value, then check with prev_index
                            if 0 <= item <= 6 and prev_index != item:
                                prev_index = item
                            else:
                                Logger.info(
                                    'Training Scheduler configuration is invalid. Ignoring pipeline %s configuration',
                                    pipeline.get('pipeline_name'))
                                continue
                        # set to activate by weekdays
                        cron_schedule.setdefault('day_of_week', ','.join(map(lambda x: str(x), weekday_list)))

                    else:
                        Logger.info(
                            'Training Scheduler configuration is invalid. Ignoring pipeline %s configuration',
                            pipeline.get('pipeline_name'))
                        continue

                else:
                    Logger.info('Training Scheduler configuration is invalid. Ignoring pipeline %s configuration',
                                pipeline.get('pipeline_name'))
                    continue

                pipeline_config.setdefault('schedule_type', 'cron')
                pipeline_config.setdefault('schedule', cron_schedule)

                self.pipelines.append(pipeline_config)


TrainingActivatorConfig = __TrainingActivatorConfig()

import asyncio
import inspect
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict

from dateutil.relativedelta import relativedelta


class ProcessPipeline(ABC):

    @staticmethod
    @abstractmethod
    def load_model(type_id, pipeline_name):
        pass

    @staticmethod
    @abstractmethod
    def save_model(pipeline, type_id, meta, model, train_results):
        pass

    @staticmethod
    @abstractmethod
    def train(type_id, config):
        pass

    @staticmethod
    @abstractmethod
    def predict(type_id, data, config):
        pass

    def get_class(self):
        return self.__class__

    def get_config(self):
        config = {}
        public_attributes = (member for member in inspect.getmembers(self) if
                             not member[0].startswith('_') and not callable(member[1]))

        for attribute in public_attributes:
            config.setdefault(attribute[0], attribute[1])

        return config

    @staticmethod
    def get_training_date_range(timedelta: relativedelta) -> object:
        """

        :param timedelta: relativedelta, the time date range required for training
        :return: start_date: datetime, end_date: datetime
        """

        end_date = datetime.now()
        start_date = end_date - timedelta
        return start_date, end_date

    def compute_training_time_delta(self, training_data_period_config: Dict):
        for field in training_data_period_config.keys():
            if training_data_period_config.get(field) != 0:
                param = {field: training_data_period_config.get(field)}
                training_time_delta = relativedelta(**param)
                return training_time_delta

    @staticmethod
    def submit_training_result(result_json, requester):
        asyncio.get_event_loop().run_until_complete(requester.post_to_submit_training_result(result_json))


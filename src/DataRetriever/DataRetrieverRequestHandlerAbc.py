from abc import ABC, abstractmethod
from datetime import datetime


class DataRetrieverRequestHandlerAbc(ABC):

    @abstractmethod
    def query_prediction_results(self, pipeline: str, type_id: int, start_date: datetime, end_date: datetime):
        pass

    @abstractmethod
    def query_all_training_model_info(self, pipeline: str, type_id: int, page_index: int = 1, page_size: int = 10):
        pass

    @abstractmethod
    def query_training_model_info(self, pipeline: str, type_id: int, date: datetime):
        pass

import json
from datetime import datetime

from lib.Common.Auxiliary import Auxiliary
from lib.Storage.StorageUtil import StorageUtil
from src.DataRetriever.DataRetrieverRequestHandlerAbc import DataRetrieverRequestHandlerAbc


class SimpleRequestHandler(DataRetrieverRequestHandlerAbc):

    def __init__(self):
        pass

    def query_prediction_results(self, pipeline: str, type_id: int, start_date: datetime, end_date: datetime):
        """

        :param pipeline: str, the unique pipeline name
        :param type_id: int, the unique ID
        :param start_date: datetime, the beginning datetime of the query
        :param end_date: datetime, the end datetime of the query
        :return: dict, { 'number_items': int, 'start_date': datetime, 'end_date': datetime, data: [{}, {}, ...]}
        """

        data = StorageUtil.read_prediction_results_by_date_range(pipeline, type_id, start_date, end_date)
        return dict(
            number_items=len(data),
            start_date=str(start_date),
            end_date=str(end_date),
            data=data
        )

    def query_all_training_model_info(self, pipeline: str, type_id: int, page_index=1, page_size: int = 10):
        """


        :param pipeline: str, the unique pipeline name
        :param type_id: int, the unique ID
        :param page_index: int, default to 1, the index of the page is requested
        :param page_size: int, default to 100, limits number of elements returned in data
        :return: dict, {
                            'number_items': int,
                            'last_completed_training_date': str(datetime),
                            'data': [str(datetime), str(datetime), ...],
                            'page_index': int,
                            'page_size': int,
                            'num_pages': int
                        }
        """
        # get all the timestamps from Storage
        timestamps = StorageUtil.get_all_saved_model_timestamps(pipeline, type_id)
        # sort the list by descending order
        timestamps.sort(reverse=True)

        timestamps = list(map(lambda x: str(x), timestamps))

        try:
            last_completed_training = timestamps[0]
        except IndexError:
            last_completed_training = None

        num_items = len(timestamps)

        page_index, num_pages, begin_index, end_index = Auxiliary.generate_pagination_info(num_items, page_index,
                                                                                           page_size)

        timestamps = timestamps[begin_index: end_index]

        rv = dict(number_items=num_items, last_completed_training_date=last_completed_training, data=timestamps,
                  page_index=page_index, num_pages=num_pages, page_size=page_size)

        return rv

    def query_training_model_info(self, pipeline: str, type_id: int, date: datetime):
        """

        :param pipeline: str, the unique pipeline name
        :param type_id: int, the unique ID
        :param date: datetime, the specific datetime that the model finished training at
        :return: dict, { 'metadata': {}, ... }
        """
        meta_json = StorageUtil.read_model_meta_by_date(pipeline, type_id, date)
        train_result_df = StorageUtil.read_training_results(type_id, pipeline, date)
        train_result_json = json.loads(train_result_df.to_json(orient='records'))

        return dict(metadata=meta_json, train_results=train_result_json)

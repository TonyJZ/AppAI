import datetime as dt
import dateutil.parser
import json
from typing import Dict, List

from lib.Config.ConfigUtil import ConfigUtil
from lib.Storage.HdfsStorage import HdfsStorage
import pandas as pd
from dateutil.relativedelta import relativedelta
from lib.Exceptions.StorageExceptions import NoModelInStorageException, IncorrectStoragePathException, \
    NoDataInStorageException


class StorageHandler():
    __DB_IO_HDFS_USER = 'root'
    _HDFS_DATA_ACCESS_PORT = '9000'
    _HDFS_WEB_ACCESS_PORT = '50070'
    __storage_handler = None

    def __init__(self, ip: str, root_folder: str):
        self.__ip = ip
        self.__root_folder = root_folder

        if self.__storage_handler is None:
            self.__storage_handler = HdfsStorage(self.__DB_IO_HDFS_USER, self.__ip, self.__root_folder)

    def _slash_join(self, *args):
        """
            Joining a list of strings by /
        :param args: a list of string
        :return:    slash-joined string
        """
        stripped_strings = []

        # striping the / in the strings
        for s in args:
            start = 0
            if s[-1] == '/':
                stripped_strings.append(s[start:-1])
            else:
                stripped_strings.append(s[start:])
        return '/'.join(stripped_strings)

    def write_raw(self, stream: str, type_id: int, date: dt.datetime, data: Dict = None):

        """Write the raw data into Big Data Storage

        Args:
            stream(str): Stream name
            type_id(int): Type ID
            date(datetime): Data received datetime
            data(dict): Data should be a dictionary

        Raises:
            Runtime error:
                date is not an instance of datetime && data is a dictionary

        Returns:
            void, raise exception if failure
        """

        if not isinstance(date, dt.datetime) or not isinstance(data, dict):
            raise

        self.__storage_handler.write_raw(stream, type_id, date, data)

    def read_raw(self, stream: str, type_id: int, date: dt.datetime):
        """Read the raw data from HDFS

        Args:
            stream(str): Stream name
            type_id(int): Type ID
            date(datetime): The date of the data looking for

        Raises:
            Runtime error:
                date is not an instance of datetime

        Returns:
            return data as Pandas.DataFrame
        """

        if not isinstance(date, dt.datetime):
            raise RuntimeError

        df = self.__storage_handler.read_raw(stream, type_id, date)

        return df

    def read_raw_duration(self, stream: str, type_id: int, time_column: str, start_time: dt.datetime, end_time: dt.datetime):
        """Read the raw data from HDFS

        Args:
            stream(str): Stream name
            type_id(int): Type ID
            start_time(datetime): The start time of the data looking for
            end_time(datetime): The end time of the data looking for

        Raises:
            Runtime error:
                date is not an instance of datetime

        Returns:
            return data as Pandas.DataFrame
        """

        if (not isinstance(start_time, dt.datetime)) or (not isinstance(end_time, dt.datetime)):
            raise RuntimeError

        all_data = pd.DataFrame()
        date = start_time
        # Todo: issue -> what if the start_time.month is bigger than end_time.month, but the year is smaller?
        while date.month <= end_time.month or date.year < end_time.year:
            one_month_data = self.__storage_handler.read_raw(stream, type_id, date)
            all_data = all_data.append(one_month_data, ignore_index=True)
            date = date + relativedelta(months=+1)

        if all_data.empty:
            return all_data

        if not isinstance(all_data[time_column][0], dt.datetime):
            for i in range(len(all_data)):
                all_data[time_column][i] = dateutil.parser.parse(all_data[time_column][i])
        
        df = all_data[(all_data[time_column] <= end_time)].reset_index(drop=True)
        df = df[(df[time_column] >= start_time)].reset_index(drop=True)
        sorted_data = df.sort_values(by=[time_column])
        return sorted_data

    def write_model(self, type_id: int, model: object, pipeline: str, date: dt.datetime):
        """ Write the trained model into Data Storage system

        Args:
            type_id(int): Unique ID
            model(object) The Keras model
            pipeline(str): pipeline name, as defined in the config
            date(datetime): training completed datetime

        Returns:
            return True if no exception raised

        """

        if not isinstance(date, dt.datetime):
            raise

        self.__storage_handler.write_model(type_id, model, pipeline, date)

        return True

    def read_model(self, type_id: int, pipeline: str, date: dt.datetime):
        """ Read the trained model from Data Storage system

        Args:
            type_id(int): Unique ID
            pipeline(str): pipeline name, as defined in the config
            date(datetime): model training completed datetime

        Returns:
            return the Keras model for successful, None otherwise

        """

        if not isinstance(date, dt.datetime):
            raise

        model = self.__storage_handler.read_model(type_id, pipeline, date)

        return model

    def write_result(self, type_id: int, date: dt.datetime, pipeline: str, data: Dict[str, any]):
        """ Save the result into Data Storage system

        Args:
            type_id(int): Unique ID
            date(datetime): date for the result data
            pipeline(str): pipeline name, as defined in the config
            data(dict): data in dict format

        Raises:
            Runtime error:
                date is not an instance of datetime

        Returns:
            return void if no exception raised

        """

        if not isinstance(date, dt.datetime):
            raise RuntimeError

        self.__storage_handler.write_result(type_id, date, pipeline, data)

    def read_result(self, type_id: int, date: dt.datetime, pipeline: str):
        """ Read the result from Data Storage system

        Args:
            type_id(int): Unique ID
            date(datetime): date for the result data
            pipeline(str): pipeline name, as defined in the config

        Raises:
            Runtime error:
                date is not an instance of datetime

        Returns:
            return data as Pandas.DataFrame

        """

        if not isinstance(date, dt.datetime):
            raise

        df = self.__storage_handler.read_result(type_id, date, pipeline)

        return df

    def write_training_results(self, type_id: int, pipeline: str, data: List[Dict[str, any]], date: dt.datetime):
        """ Save the training result into Data Storage system

        Args:
            type_id(int): Unique ID
            pipeline(str): pipeline name, as defined in the config
            data(dict): data in dict format
            date(datetime): model training completed datetime

        Raises:

        Returns:
            return void if no exception raised

        """

        if not isinstance(date, dt.datetime):
            raise

        self.__storage_handler.write_training_results(type_id, pipeline, data, date)

    def read_training_results(self, type_id: int, pipeline: str, date: dt.datetime):
        """ Read the training result from Data Storage system

        Args:
            type_id(int): Unique ID
            pipeline(str): pipeline name, as defined in the config
            date(datetime): model training completed datetime

        Raises:
            Runtime error:
                date is not an instance of datetime
        Returns:
            return data as Pandas.DataFrame

        """

        if not isinstance(date, dt.datetime):
            raise

        df = self.__storage_handler.read_training_results(type_id, pipeline, date)

        return df

    def write_model_meta(self, type_id: int, pipeline: str, data: Dict[str, any], date: dt.datetime):
        """ Save the model meta into Data Storage system

        Args:
            type_id(int): Unique ID
            pipeline(str): pipeline name, as defined in the config
            data(dict): data in dict format
            date(datetime): the date when the training completed

        Raises:
            Runtime error:
                date is not an instance of datetime

        Returns:
            return True if no exception raised

        """

        if not isinstance(date, dt.datetime):
            raise

        self.__storage_handler.write_model_meta(type_id, pipeline, data, date)
        return True

    def read_model_meta(self, type_id: int, pipeline: str, date: dt.datetime):
        """ Read the model meta from Data Storage system

        Args:
            type_id(int): Unique ID
            pipeline(str): pipeline name, as defined in the config

        Raises:
            Runtime error:
                date is not an instance of datetime

        Returns:
            return data as Pandas.DataFrame

        """

        if not isinstance(date, dt.datetime):
            raise

        df = self.__storage_handler.read_model_meta(type_id, pipeline, date)

        return df

    def check_file_exists(self, path: str, filename: str):
        """

        :param path: path of the file
        :param filename: the file name
        :return: True/False
        """
        target = self._slash_join(path, filename)
        return self.__storage_handler.check_file_exists(target)

    # Delete file
    def delete_file(self, file: str):
        """

        :param file: the file (contains full path) that needs to be deleted
        :return: True/False
        """
        return self.__storage_handler.delete_content(file)

    def delete_folder(self, folder: str):
        """

        :param folder: the folder (contains full path) that needs to be deleted
        :return:
        """
        return self.__storage_handler.delete_content(folder, is_folder=True)

    def set_stream_headers(self, stream: str, headers: List[str]):
        """

        :param stream: the unique stream name
        :param headers: the raw data headers
        :return: void
        """

        self.__storage_handler.set_stream_headers(stream, headers)

    def set_pipeline_headers(self, pipeline: str, result_headers: List[str], model_meta_headers: List[str],
                             training_result_header: List[str]):
        """

        :param pipeline: the unique pipeline name
        :param result_headers: the prediction result headers
        :param model_meta_headers: the model meta headers
        :param training_result_header: the model training result headers
        :return:
        """
        self.__storage_handler.set_pipeline_headers(pipeline, result_headers, model_meta_headers,
                                                    training_result_header)

    def get_pipeline_headers(self, pipeline: str):
        """

        :param pipeline: the unique pipeline name
        :return: a list of strings
        """
        return self.__storage_handler.get_pipeline_headers(pipeline)

    def read_model_and_meta(self, pipeline: str, type_id: int):
        """

        :param pipeline: the unique pipeline name
        :param type_id: the unique type ID
        :return:
        """

        # get the latest trained model
        latest_date = self.__storage_handler.get_latest_model_folder(type_id, pipeline)

        # returns Model object and Model Meta
        model = self.read_model(type_id, pipeline, latest_date)
        meta = self.read_model_meta(type_id, pipeline, latest_date)

        return model, meta

    def write_model_and_meta(self, pipeline: str, type_id: int, model: object, meta: Dict[str, any],
                             training_completed_date: dt.datetime):
        """

        :param pipeline: the unique pipeline name
        :param type_id: the unique type ID
        :param model: the model needs to be stored
        :param meta: the model metadata
        :param training_completed_date: the model training completed datetime
        :return: void
        """
        self.write_model_meta(type_id, pipeline, meta, training_completed_date)
        self.write_model(type_id, model, pipeline, training_completed_date)

    def get_all_saved_model_timestamps(self, pipeline: str, type_id: int):
        """

        :param pipeline: the unique pipeline name
        :param type_id: the unique type ID
        :return: a list of datetime object
        """
        return self.__storage_handler.get_all_model_create_date(type_id, pipeline)

    def read_model_meta_by_date(self, pipeline: str, type_id: int, date: dt.datetime):
        """

        :param pipeline: the unique pipeline name
        :param type_id: the unique type ID
        :param date: the target date
        :return: meta: dict
        """

        # returns Model Meta
        meta_df = self.read_model_meta(type_id, pipeline, date)

        # meta data should only has one row
        meta_json = json.loads(meta_df.to_json(orient='records'))[0]

        return meta_json

    def read_prediction_results_by_date_range(self, pipeline: str, type_id: int, begin_date: dt.datetime,
                                              end_date: dt.datetime):
        """

        :param pipeline: the unique pipeline name
        :param type_id: the unique type id
        :param begin_date: the begin date
        :param end_date: the end date
        :return: list[dict]
        """

        date_list = self.__storage_handler.get_date_list_to_read_results(begin_date, end_date)

        # get the first date and perform filtering
        data_df = self.read_result(type_id, date_list[0], pipeline)
        if not data_df.empty:
            data_df = data_df[data_df['create_date'] > str(begin_date)]
        for date in date_list[1 : -1]:
            data_df = data_df.append(self.read_result(type_id, date, pipeline))

        # get the first date and perform filtering
        last_data_df = self.read_result(type_id, date_list[-1], pipeline)
        if not last_data_df.empty:
            last_data_df = last_data_df[last_data_df['create_date'] < str(end_date)]
            data_df = data_df.append(last_data_df)
        
        return json.loads(data_df.to_json(orient='records'))


__project_root_folder = '/cnaf-sagw'
# load configuration
__hdfs_ip = ConfigUtil.get_value_by_key('resources', 'hdfs', 'hdfs_host_ip')
__stream_configs = ConfigUtil.get_value_by_key('streams')

# setup singleton StorageUtil
StorageUtil = StorageHandler(__hdfs_ip, __project_root_folder)

# Configure Stream to Headers mapping
# loop through config to set up stream to headers relationship
for stream_config in __stream_configs:
    stream_key = __stream_configs.get(stream_config).get('stream')
    stream_raw_headers = __stream_configs.get(stream_config).get('headers')
    StorageUtil.set_stream_headers(stream_key, stream_raw_headers)

    # Configure pipeline to other Headers mapping
    for pipeline_config in __stream_configs.get(stream_config).get('pipelines'):
        StorageUtil.set_pipeline_headers(pipeline_config.get('pipeline_name'),
                                         pipeline_config.get('result_headers'),
                                         pipeline_config.get('model_meta_headers'),
                                         pipeline_config.get('training_result_headers'))

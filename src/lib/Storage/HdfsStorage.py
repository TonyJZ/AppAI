import _pickle as pickle
import csv
import threading
from collections import deque
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta
from hdfs import InsecureClient, HdfsError
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json

from lib.Exceptions.StorageExceptions import NoModelInStorageException, IncorrectStoragePathException


class HdfsStorage(object):
    _csv_suffix = '.csv'
    _raw_dir = 'raw'
    _raw_data_filename_template = '%Y-%m'
    _result_dir = 'results'
    _result_data_filename_template = '%Y-%m'

    _model_arch_suffix = '.json'
    _model_weights_suffix = '.h5'
    _model_dir = 'models'

    _model_date_folder_template = "%Y%m%d_%H%M%S.%f"
    _model_arch_file = 'arch' + _model_arch_suffix
    _model_weights_file = 'weights' + _model_weights_suffix
    _model_meta_file = 'meta' + _csv_suffix
    _training_result_file = 'train_results' + _csv_suffix

    # root_dir/raw_dir/stream/type_id/filename.csv
    _raw_location_template = '{}/' + _raw_dir + '/{}/{}/{}' + _csv_suffix
    # root_dir/result_dir/pipeline/type_id/filename.csv
    _result_location_template = '{}/' + _result_dir + '/{}/{}/{}' + _csv_suffix

    # root_dir/model_dir/pipeline/type_id/yyyymmdd_HHMMSS.ssssss_Zone/...files
    _model_location_template = '{}/' + _model_dir + '/{}/{}/{}/{}'

    _result_headers_key = 'result_headers'
    _model_meta_headers_key = 'model_meta'
    _training_result_headers_key = 'training_result'

    def __init__(self, hdfs_user, ip, root_dir, web_access_port='50070',
                 data_access_port='9000', max_concurrent_clients=10):
        self.user = hdfs_user
        self.root_dir = root_dir

        self.ip = ip
        self.web_access_port = web_access_port
        self.data_access_port = data_access_port

        self.web_url = 'http://' + self.__ip_port_join(self.ip, self.web_access_port)
        self.data_url = 'hdfs://' + self.__ip_port_join(self.ip, self.data_access_port)

        self.clients = []
        self.client_indexes = deque(maxlen=max_concurrent_clients)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

        self._lock.acquire()
        for x in range(max_concurrent_clients):
            client = InsecureClient(self.web_url, self.user)
            self.clients.append(client)
            self.client_indexes.append(x)

        self._lock.release()

        self._stream_to_headers = {}
        # pipeline -> storage_type {result, model_meta} -> headers
        self._pipeline_to_type_to_headers = {}
        
        return 

    def __del__(self):
        # The cleanup is not really working due to the execution time of the __del__ method
        
        for idx in self.client_indexes:
            # print('Free resource for client[{}]'.format(idx))
            try:
                client = self.clients[idx]
                if client._session is None:
                    continue
                else:
                    client._session.close()
            except Exception as e:
                print(e)
                # sometimes the client object is free in advacnce                    
                continue
        print('Clean up HDFS Storage client resources')
        return

    def __ip_port_join(self, ip, port):
        """Joining two strings by : """
        return ':'.join([ip, port])

    def _get_client(self):
        with self._cv:
            while len(self.client_indexes) == 0:
                self._cv.wait()
            client_idx = self.client_indexes.popleft()
            # print('_get_client: index {}'.format(client_idx))

        hdfs_client = self.clients[client_idx]
        return client_idx, hdfs_client

    def _putback_client(self, idx):
        with self._cv:
            self.client_indexes.append(idx)
            self._cv.notifyAll()

    # def _validate_solution_type(self, solution_type):
    #     if type(solution_type) != int \
    #             or solution_type < 0 \
    #             or solution_type >= len(HdfsStorage._solution_type_dir):
    #         raise

    def _create_directory(self, hdfs_client, path):
        hdfs_client.makedirs(path)

    def _list_contents_in_path(self, path=''):
        '''List contents inside the path
        
        Args:
            path (str): path check
        
        Returns:
            list: list of file/folder names
        '''

        if path is '':
            print('Incorrect input')
            raise IncorrectStoragePathException('incorrect path[{}]'.format(path))
        # obtain the client connection from object
        client_idx, hdfs_client = self._get_client()

        try:
            return hdfs_client.list(path, status=True)
        except HdfsError as err:
            if 'not a directory' in err.message:
                return None
            elif 'does not exist' in err.message:
                # create the directory
                self._create_directory(hdfs_client, path)
                return []
            else:
                raise err
        finally:
            self._putback_client(client_idx)

    def check_file_exists(self, full_file=''):

        if full_file is '':
            print('Incorrect input')
            raise

        # separate filename and filepath from full_file
        split_by_slash = full_file.split('/')
        path = '/'.join(split_by_slash[:-1])
        filename = split_by_slash[-1]
        content_list = self._list_contents_in_path(path)
        found = list(filter(lambda x: x[0] == filename, content_list))

        if found is None or len(found) == 0:
            return False
        else:
            for item in found:
                if item[1]['type'] == 'FILE':
                    return True
            return False

    def _append_csv(self, full_file, data):
        # obtain the client connection from object
        client_idx, hdfs_client = self._get_client()
        try:
            with hdfs_client.write(full_file, encoding='utf-8', append=True) as writer:
                outfile = csv.writer(writer)
                outfile.writerow(data)
        except Exception as err:
            raise err

        finally:
            self._putback_client(client_idx)

    def _write_csv(self, full_file, headers, data, overwrite_file=False):
        fileExist = self.check_file_exists(full_file) if not overwrite_file else False
        if fileExist:
            # append
            self._append_csv(full_file, data)
        else:
            # obtain the client connection from object
            client_idx, hdfs_client = self._get_client()
            try:
                # save data to file using csv writer
                with hdfs_client.write(full_file, encoding='utf-8', overwrite=overwrite_file) as writer:
                    outfile = csv.writer(writer)
                    outfile.writerow(headers)
                    # check to see if the contents in the data is an list or not
                    if isinstance(data[0], list):
                        # contents in the data is a list, means each item in data is one row
                        outfile.writerows(data)
                    else:
                        # contents in the data is not a list, means each item is one column
                        outfile.writerow(data)
            except Exception as err:
                raise err

            finally:
                self._putback_client(client_idx)

    def write_raw(self, stream, type_id, date, data=None):
        filename = date.strftime(HdfsStorage._raw_data_filename_template)
        file_to_save = HdfsStorage._raw_location_template.format(self.root_dir, stream, type_id, filename)

        # transform the dict to list
        headers = self._stream_to_headers.get(stream)
        data_list = self.__transform_dict_val_to_list(headers, data)
        self._write_csv(file_to_save, headers, data_list)

        return True

    def _read_csv(self, full_file):
        data = None
        if self.check_file_exists(full_file):
            # read the file
            # obtain the client connection from object
            client_idx, hdfs_client = self._get_client()
            try:

                # save data to file using csv writer
                with hdfs_client.read(full_file, encoding='utf-8') as reader:
                    data = pd.read_csv(reader, header=0)

            except Exception as err:
                raise err

            finally:
                self._putback_client(client_idx)

        if data is None:
            data = pd.DataFrame()

        return data

    def read_raw(self, stream, type_id, date):

        filename = date.strftime(HdfsStorage._raw_data_filename_template)
        file_to_read = HdfsStorage._raw_location_template.format(self.root_dir, stream, type_id, filename)

        return self._read_csv(file_to_read)

    def write_result(self, type_id, date, pipeline, data):
        # self._validate_solution_type(solution_type)

        filename = date.strftime(HdfsStorage._result_data_filename_template)
        file_to_save = HdfsStorage._result_location_template.format(self.root_dir,
                                                                    pipeline,
                                                                    type_id, filename)
        headers = self.get_pipeline_headers(pipeline, self._result_headers_key)
        data_list = self.__transform_dict_val_to_list(headers, data)
        self._write_csv(file_to_save, headers, data_list)

    def read_result(self, type_id, date, pipeline):

        filename = date.strftime(HdfsStorage._result_data_filename_template)
        file_to_read = HdfsStorage._result_location_template.format(self.root_dir,
                                                                    pipeline,
                                                                    type_id, filename)

        return self._read_csv(file_to_read)

    def write_training_results(self, type_id, pipeline, data, date):

        file_to_save = self._get_model_path(type_id, pipeline, date).format(HdfsStorage._training_result_file)

        headers = self.get_pipeline_headers(pipeline, self._training_result_headers_key)
        data_list = []
        for item in data:
            data_list.append(self.__transform_dict_val_to_list(headers, item))
        self._write_csv(file_to_save, headers, data_list)

    def read_training_results(self, type_id, pipeline, date):

        file_to_read = self._get_model_path(type_id, pipeline, date).format(HdfsStorage._training_result_file)

        return self._read_csv(file_to_read)

    def write_model_meta(self, type_id, pipeline, data, date):

        file_to_save = self._get_model_path(type_id, pipeline, date).format(HdfsStorage._model_meta_file)
        headers = self.get_pipeline_headers(pipeline, self._model_meta_headers_key)
        data_list = self.__transform_dict_val_to_list(headers, data)
        self._write_csv(file_to_save, headers, data_list, overwrite_file=True)

    def read_model_meta(self, type_id, pipeline, date):

        file_to_read = self._get_model_path(type_id, pipeline, date).format(HdfsStorage._model_meta_file)

        return self._read_csv(file_to_read)

    def _get_model_path(self, type_id, pipeline, date):
        date_folder = date.strftime(HdfsStorage._model_date_folder_template)
        return HdfsStorage._model_location_template.format(self.root_dir,
                                                           pipeline, type_id, date_folder, {})

    def _generate_model_filenames(self, type_id, pipeline, date):

        arch_to_save = self._get_model_path(type_id, pipeline, date).format(HdfsStorage._model_arch_file)
        param_to_save = self._get_model_path(type_id, pipeline, date).format(HdfsStorage._model_weights_file)

        return arch_to_save, param_to_save

    def write_model(self, type_id, model, pipeline, date):

        arch_filename, param_filename = self._generate_model_filenames(type_id, pipeline, date)

        self._write_model(arch_filename, param_filename, model)

    def _write_model(self, arch_file, param_file, model):
        # obtain the client connection from object
        client_idx, hdfs_client = self._get_client()
        try:
            with hdfs_client.write(arch_file, encoding='utf-8', overwrite=True) as writer:
                writer.write(model.to_json())

            with hdfs_client.write(param_file, overwrite=True) as writer:
                writer.write(pickle.dumps(model.get_weights()))

        except Exception as err:
            raise err

        finally:
            self._putback_client(client_idx)

    def read_model(self, type_id, pipeline, date):

        arch_file, param_file = self._generate_model_filenames(type_id, pipeline, date)
        model = self._read_model(arch_file, param_file)

        return model

    def _read_model(self, arch_file, param_file):
        model = None
        # obtain the client connection from object
        client_idx, hdfs_client = self._get_client()
        try:
            K.clear_session()
            with hdfs_client.read(arch_file, encoding='utf-8') as reader:
                model = model_from_json(reader.read())

            with hdfs_client.read(param_file) as reader:
                model_weights = pickle.loads(reader.read())

            model.set_weights(model_weights)

        except Exception as err:
            raise err

        finally:
            self._putback_client(client_idx)

        return model

    def get_latest_model_folder(self, type_id, pipeline):
        """

        :param type_id: the unique ID
        :param pipeline: the pipeline name
        :return: model folder in datetime format
        """
        path = HdfsStorage._model_location_template.format(self.root_dir, pipeline, type_id, '', '')
        date_folders = self._list_contents_in_path(path)

        if 0 == len(date_folders):
            raise NoModelInStorageException('No model saved for pipeline[{}] typeId[{}]'.format(pipeline, type_id))

        date_folders.sort(reverse=True)
        # return the folder name only
        return self.get_date_from_date_folder(date_folders[0][0])

    def get_date_from_date_folder(self, date_folder_name):
        return datetime.strptime(date_folder_name, HdfsStorage._model_date_folder_template)

    def get_all_model_create_date(self, type_id, pipeline):
        """

        :param type_id: the unique ID
        :param pipeline: the pipeline name
        :return: a list of model folder in datetime format
        """
        path = HdfsStorage._model_location_template.format(self.root_dir, pipeline, type_id, '', '')
        date_folders = self._list_contents_in_path(path)
        folder_name_list = list(map(lambda x: self.get_date_from_date_folder(x[0]), date_folders))
        return folder_name_list

    def delete_content(self, file_path, is_folder=False):

        # obtain the client connection from object
        client_idx, hdfs_client = self._get_client()

        try:
            return hdfs_client.delete(file_path, recursive=is_folder)
        except HdfsError as err:
            raise err
        finally:
            self._putback_client(client_idx)

    def set_stream_headers(self, stream, headers):
        self._stream_to_headers[stream] = headers

    def set_pipeline_headers(self, pipeline, result_headers, model_meta_headers, training_result_header):

        if result_headers is None:
            result_headers = self.get_pipeline_headers(pipeline, self._result_headers_key)

        if model_meta_headers is None:
            model_meta_headers = self.get_pipeline_headers(pipeline, self._model_meta_headers_key)

        if training_result_header is None:
            training_result_header = self.get_pipeline_headers(pipeline, self._training_result_headers_key)

        pipeline_dict = {self._result_headers_key: result_headers, self._model_meta_headers_key: model_meta_headers,
                         self._training_result_headers_key: training_result_header}
        self._pipeline_to_type_to_headers[pipeline] = pipeline_dict

    def get_pipeline_headers(self, pipeline, type):
        solution_dict = self._pipeline_to_type_to_headers.get(pipeline)
        return solution_dict.get(type)

    def get_date_list_to_read_results(self, begin_date: datetime, end_date: datetime):

        date_list = []
        while begin_date.strftime(HdfsStorage._result_data_filename_template) <= end_date.strftime(
                HdfsStorage._result_data_filename_template):
            date_list.append(begin_date)
            begin_date = begin_date + relativedelta(months=+1)

        return date_list

    @staticmethod
    def __transform_dict_val_to_list(headers, data):
        data_list = []
        for header in headers:
            data_value = data.get(header)
            if data_value is None:
                data_list.append('')
            else:
                data_list.append(data_value)

        return data_list

import csv
import pickle
from datetime import datetime
from unittest import TestCase

import numpy as np
import pandas as pd
import tensorflow.keras as keras
from hdfs import InsecureClient
from pandas.util.testing import assert_frame_equal

load_model = keras.models.load_model
Sequential = keras.models.Sequential
LSTM = keras.layers.LSTM
Dense = keras.layers.Dense

from lib.Storage.StorageUtil import StorageHandler

__hdfs_ip = '10.10.10.184'
__user = 'root'
__web_url = 'http://' + __hdfs_ip + ':50070'

_hdfs_root_folder = '/unittest'
_storage_util = StorageHandler(__hdfs_ip, _hdfs_root_folder)
_storage_util.set_stream_headers('stream1', ['a', 'b', 'c'])
_pipeline = 'prediction'
_storage_util.set_pipeline_headers(_pipeline, ['ra', 'rb', 'rc'], ['min', 'max'], ['ta', 'tb', 'tc'])
_file_to_create = _hdfs_root_folder + '/test/test.txt'
_client = InsecureClient(__web_url, __user)

_type_id = 9999
_raw_data_file_path = _hdfs_root_folder + '/raw/stream1/9999/2019-01.csv'
_raw_data = [['a', 'b', 'c'], [1, 2, 3]]

_result_data_file_path = _hdfs_root_folder + '/results/prediction/9999/2019-01.csv'
_result_data = [['ra', 'rb', 'rc'], [1, 2, 3]]

_now = datetime.now()
_now_str = _now.strftime("%Y%m%d_%H%M%S.%f")
_training_result_data_file_path = _hdfs_root_folder + '/models/prediction/{}/{}/train_results.csv'.format(_type_id, _now_str)
_training_result_data = [['ta', 'tb', 'tc'], [1, 2, 3], [4, 5, 6]]

_model_meta_data_file_path = _hdfs_root_folder + '/models/prediction/9999_meta.csv'
_model_meta_data = [['min', 'max'], [1, 2]]

_arch_file_path = _hdfs_root_folder + '/models/prediction/9999.json'
_param_file_path = _hdfs_root_folder + '/models/prediction/9999.h5'
_model_arch = None
_model_param = None


class TestStorageHandlerReadFuncs(TestCase):

    def setUp(self) -> None:
        # prepare raw
        with _client.write(_raw_data_file_path, encoding='utf-8', overwrite=True) as writer:
            outfile = csv.writer(writer)
            outfile.writerows(_raw_data)

        # prepare result
        with _client.write(_result_data_file_path, encoding='utf-8', overwrite=True) as writer:
            outfile = csv.writer(writer)
            outfile.writerows(_result_data)

        # prepare training result
        with _client.write(_training_result_data_file_path, encoding='utf-8') as writer:
            outfile = csv.writer(writer)
            outfile.writerows(_training_result_data)

        # prepare model meta
        # with _client.write(_model_meta_data_file_path, encoding='utf-8', overwrite=True) as writer:
        #     outfile = csv.writer(writer)
        #     outfile.writerows(_model_meta_data)

        # keras.backend.clear_session()
        # _model = Sequential()
        # _model.add(LSTM(20, activation="relu", input_shape=(128, 50), kernel_initializer='random_uniform'))
        # _model.add(Dense(1, kernel_initializer='random_uniform', activation='linear'))
        # _adam_optimizer = keras.optimizers.Adam(lr=0.001)
        # _model.compile(optimizer=_adam_optimizer, loss='mse')

        # prepare model
        # with _client.write(_arch_file_path, encoding='utf-8', overwrite=True) as writer:
        #     writer.write(_model.to_json())
        #     global _model_arch
        #     _model_arch = _model.to_json()
        #
        # with _client.write(_param_file_path, overwrite=True) as writer:
        #     writer.write(pickle.dumps(_model.get_weights()))
        #     global _model_param
        #     _model_param = _model.get_weights()

    def tearDown(self) -> None:
        _client.delete(_hdfs_root_folder, recursive=True)

    def test_read_raw(self):
        print("\n\n*********\nunit test for read_raw")
        target_date = datetime.now().replace(year=2019, month=1, day=1)
        data = _storage_util.read_raw('stream1', 9999, target_date)
        expect_data = pd.DataFrame(np.array([_raw_data[1]]), columns=_raw_data[0])
        assert_frame_equal(expect_data, data)

    def test_read_result(self):
        print("\n\n*********\nunit testing read_result")
        target_date = datetime.now().replace(year=2019, month=1, day=1)
        data = _storage_util.read_result(9999, target_date, _pipeline)
        expect_data = pd.DataFrame(np.array([_result_data[1]]), columns=_result_data[0])
        assert_frame_equal(expect_data, data)

    def test_read_training_results(self):
        print("\n\n*********\nunit testing read_training_results")
        data = _storage_util.read_training_results(_type_id, _pipeline, _now)
        expect_data = pd.DataFrame(_training_result_data[1:], columns=_training_result_data[0])
        assert_frame_equal(expect_data, data)

    # def test_read_model_meta(self):
    #     print("\n\n*********\nunit testing read_model_meta")
    #     data = _storage_util.read_model_meta(9999, _pipeline)
    #     expect_data = pd.DataFrame(np.array([_model_meta_data[1]]), columns=_model_meta_data[0])
    #     assert_frame_equal(expect_data, data)
    #
    # def test_read_model(self):
    #     print('\n\n*********\nUnit Test Read Model from HDFS')
    #     model = _storage_util.read_model(9999, _pipeline)
    #     self.assertIsNotNone(model, msg='model is None')
    #     self.assertEqual(_model_arch, model.to_json())
    #     # # Haven't find a way to compare the weights yet.
    #     # self.assertEqual(len(_model_param), len(model.get_weights()))

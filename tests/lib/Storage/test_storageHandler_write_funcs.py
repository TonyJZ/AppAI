from datetime import datetime
from unittest import TestCase

import tensorflow.keras as keras
from hdfs import InsecureClient
import time

load_model = keras.models.load_model
Sequential = keras.models.Sequential
LSTM = keras.layers.LSTM
Dense = keras.layers.Dense

from lib.Storage.StorageUtil import StorageHandler

__hdfs_ip = '10.10.10.111'
__user = 'root'
__web_url = 'http://' + __hdfs_ip + ':50070'

_hdfs_root_folder = '/unittest'
_storage_util = StorageHandler(__hdfs_ip, _hdfs_root_folder)
_storage_util.set_stream_headers('stream1', ['a', 'b', 'c'])
_pipeline = 'prediction'
_storage_util.set_pipeline_headers(_pipeline, ['ra', 'rb', 'rc'], ['min', 'max'], ['ta', 'tb', 'tc'])

_file_to_create = _hdfs_root_folder + '/test/test.txt'
_client = InsecureClient(__web_url, __user)


class TestStorageHandlerWriteFuncs(TestCase):

    def setUp(self) -> None:
        _client.delete(_hdfs_root_folder, recursive=True)

    def tearDown(self) -> None:
        _client.delete(_hdfs_root_folder, recursive=True)

    def test_write_raw(self):
        data = {
            'a': 1,
            'b': 2,
            'c': 3
        }
        _storage_util.write_raw('stream1', 9999, datetime.now(), data)

        self.assertTrue(True)

    def test_write_raw_one_empty_field(self):
        data = {
            'a': 1,
            'c': 3
        }
        _storage_util.write_raw('stream1', 9999, datetime.now(), data)

        self.assertTrue(True)

    def test_write_raw_date_error(self):
        print("\n\n*********\nunit testing write_raw error")
        data = {
            'a': 1,
            'b': 2,
            'c': 3
        }
        with self.assertRaises(RuntimeError) as exception:
            _storage_util.write_raw('stream1', 9999, None, data)

    def test_write_result(self):
        print("\n\n*********\nunit testing write_result")
        data = {
            'ra': 1,
            'rb': 2,
            'rc': 3
        }
        _storage_util.write_result(9999, datetime.now(), _pipeline, data)

        self.assertTrue(True)

    def test_write_training_results(self):
        print("\n\n*********\nunit testing write_training_results")
        data = [{
            'ta': 1,
            'tb': 2,
            'tc': 3
        }, {
            'ta': 4,
            'tb': 5,
            'tc': 6
        }
        ]
        _storage_util.write_training_results(9999, _pipeline, data, datetime.now())

        self.assertTrue(True)

    # def test_write_model_error(self):
    #     print("\n\n*********\nunit testing write_model error")
    #     with self.assertRaises(Exception) as exception:
    #         _storage_util.write_model(9999, None, 3)
    #
    # def test_write_model_meta(self):
    #     print("\n\n*********\nun4it testing write_model_meta")
    #     data = {
    #         'min': 10,
    #         'max': 200,
    #     }
    #     rv = _storage_util.write_model_meta(9999, _pipeline, data)
    #
    #     self.assertTrue(rv)
    #
    # def test_write_model(self):
    #     print('\n\n*********\nUnit Test Write Model to HDFS')
    #
    #     model = Sequential()
    #     model.add(LSTM(20, activation="relu", input_shape=(128, 50), kernel_initializer='random_uniform'))
    #     model.add(Dense(1, kernel_initializer='random_uniform', activation='linear'))
    #     adam_optimizer = keras.optimizers.Adam(lr=0.001)
    #     model.compile(optimizer = adam_optimizer, loss='mse')
    #
    #     rv = _storage_util.write_model(9999, model, _pipeline)
    #
    #     self.assertTrue(rv)


if __name__ == '__main__':
    data = {
            'a': 1,
            'b': 2,
            'c': 3
        }

    while True:
        try:
            _storage_util.write_raw('stream1', 9999, datetime.now(), data)
            # time.sleep(0.1)
        except Exception as e:
            print(e)
    
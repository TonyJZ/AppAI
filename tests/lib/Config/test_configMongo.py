import json
import unittest

from mock import Mock

from lib.Config.ConfigMongo import ConfigMongo


class TestConfigMongo(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_get_all_config(self):
        # Todo: incomplete unit test
        config_handler = ConfigMongo()

        # expected_value = 'cnaf-sagw'

        value = config_handler.get_all_config()
        print('unittest: ', value)
        # self.assertEqual(expected_value, value)

    def test_get_value_by_key(self):
        # Todo: incomplete unit test
        from lib.Common.Auxiliary import Auxiliary
        Auxiliary.get_mongo_config_server_info = Mock(return_value = dict(ip='10.10.10.111', port=27017, db='cnaf-sagw'))
        print(Auxiliary.get_mongo_config_server_info())
        config_handler = ConfigMongo()
        expected_value = 'cnaf-sagw'

        value = config_handler.get_value_by_key('streams')
        print('unittest: ', value)
        # self.assertEqual(expected_value, value)


if __name__ == '__main__':
    unittest.main()

import json
import unittest

from lib.Config.ConfigLocal import ConfigLocal

class TestConfigLocal(unittest.TestCase):

    def setUp(self) -> None:
        # mocking the path to the configuration file
        ConfigLocal.file_path = './config.json'

    def tearDown(self) -> None:
        pass

    def test_get_json(self):
        config_handler = ConfigLocal()
        expected_json_str = '{\
                                    "lvl1-obj": {\
                                        "lvl2-obj": {\
                                            "lvl3-obj": {\
                                                "key1": "value1",\
                                                "key2": "value2"\
                                            }\
                                        },\
                                        "lvl2-arr": [ "arr-item1", "arr-item2", "arr-item3" ]\
                                    }\
                                }'
        expected_json = json.loads(expected_json_str)
        json_config = config_handler.get_json()
        self.assertEqual(expected_json, json_config)

    def test_get_collection(self):
        config_handler = ConfigLocal()

        expected_json_str = '{\
                                "lvl2-obj": {\
                                    "lvl3-obj": {\
                                        "key1": "value1",\
                                        "key2": "value2"\
                                    }\
                                },\
                                "lvl2-arr": [ "arr-item1", "arr-item2", "arr-item3" ]\
                            }'
        expected_json_collection = json.loads(expected_json_str)

        collection = config_handler.get_collection('lvl1-obj')
        self.assertEqual(expected_json_collection, collection)

    def test_get_collection_non_exist_collection_key(self):
        config_handler = ConfigLocal()

        collection = config_handler.get_collection('not-exist')
        self.assertIsNone(collection)

    def test_get_field(self):
        config_handler = ConfigLocal()

        expected_value = ['arr-item1', 'arr-item2', 'arr-item3']

        value = config_handler.get_field('lvl1-obj', 'lvl2-arr')
        self.assertEqual(expected_value, value)

    def test_get_all_config(self):
        config_handler = ConfigLocal()
        expected_json_str = '{\
                                            "lvl1-obj": {\
                                                "lvl2-obj": {\
                                                    "lvl3-obj": {\
                                                        "key1": "value1",\
                                                        "key2": "value2"\
                                                    }\
                                                },\
                                                "lvl2-arr": [ "arr-item1", "arr-item2", "arr-item3" ]\
                                            }\
                                        }'
        expected_json = json.loads(expected_json_str)
        json_config = config_handler.get_all_config()
        self.assertEqual(expected_json, json_config)


    def test_get_value_by_key(self):
        config_handler = ConfigLocal()

        expected_value = 'value1'

        value = config_handler.get_value_by_key('lvl1-obj', 'lvl2-obj', 'lvl3-obj', 'key1')
        self.assertEqual(expected_value, value)


    def test_get_value_by_key_non_exist_keys_expected_exception(self):
        config_handler = ConfigLocal()

        with self.assertRaises(AttributeError) as exception:
            value = config_handler.get_value_by_key('lvl1-obj', 'non-exist', 'non-exist2')


if __name__ == '__main__':
    unittest.main()

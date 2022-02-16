from pymongo import MongoClient

from lib.Common.Auxiliary import Auxiliary
from lib.Config.ConfigAbc import ConfigAbc


class ConfigMongo(ConfigAbc):

    __config_section_key = 'config_section'

    def __init__(self):
        server_info = Auxiliary.get_mongo_config_server_info()
        self.__client = MongoClient(server_info.get('ip'), server_info.get('port'))
        self.__collection = self.__client.get_database(server_info.get('db')).get_collection('config')

        if self.__client is None or self.__collection is None:
            raise Exception('Failed to build the Mongo Client')

    def get_all_config(self):
        docs = self.__collection.find()
        config = {}

        for doc in docs:
            doc.pop('_id', None)
            config.setdefault(doc.get(self.__config_section_key), doc)

        return config

    def get_value_by_key(self, *args):
        find_condition = {}
        find_condition.setdefault(self.__config_section_key, args[0])

        doc = self.__collection.find_one(find_condition)
        doc.pop('_id', None)
        value = doc

        for arg in args[1:]:
            value = value.get(arg)

        return value

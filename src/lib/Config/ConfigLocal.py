import json

from .ConfigAbc import ConfigAbc


class ConfigLocal(ConfigAbc):
    # Note: can only be called from .../AppAI/
    file_path = 'src/lib/Config/config.json'

    def __init__(self):
        self.config = self.get_json()

    def get_json(self):
        with open(self.file_path) as json_file:
            data = json.load(json_file)
            return data

    def get_collection(self, collection_name):

        collection = self.config.get(collection_name)
        return collection

    def get_field(self, collection_name, field_name):

        collection = self.config.get(collection_name)
        if collection is None:
            return None
        value = collection.get(field_name)
        return value

    def get_all_config(self):
        return self.config

    def get_value_by_key(self, *args):
        value = self.config
        for arg in args:
            value = value.get(arg)

        return value
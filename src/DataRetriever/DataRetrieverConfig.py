from lib.Config.ConfigUtil import ConfigUtil

_CONFIG_SERVICE_KEY = 'services'
_CONFIG_DATA_RETRIEVER_KEY = 'DataRetriever'


class __DataRetrieverConfig:

    def __init__(self):
        self.web_api_port = ConfigUtil.get_value_by_key(_CONFIG_SERVICE_KEY, _CONFIG_DATA_RETRIEVER_KEY,
                                                        'web_api_port')


DataRetrieverConfig = __DataRetrieverConfig()

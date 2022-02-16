from lib.Log.Log import Logger
from src.DataRetriever.DataRetrieverConfig import DataRetrieverConfig
from src.DataRetriever.DataRetrieverRequestHandler import DataRetrieverRequestHandler
from src.DataRetriever.DataRetrieverWeb import DataRetrieverWeb


class DataRetriever:

    def __init__(self):
        Logger.info('Initializing Data Retriever')
        self.request_handler = DataRetrieverRequestHandler()
        self.web_app = DataRetrieverWeb(self.request_handler)

    def run(self):
        Logger.info('Web API port: {}'.format(DataRetrieverConfig.web_api_port))
        self.web_app.start(threaded=True, host="0.0.0.0", port=DataRetrieverConfig.web_api_port)


if __name__ == '__main__':
    Logger.info('Starting the Data Retriever Service')
    data_retriever = DataRetriever()
    data_retriever.run()

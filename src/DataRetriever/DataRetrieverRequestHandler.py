from abc import ABC, abstractmethod
from datetime import datetime
from src.DataRetriever.SimpleRequestHandler import SimpleRequestHandler as __Simple_Request_Handler

__HANDLER_MAP = {
    'simple': __Simple_Request_Handler,
}

__USE_HANDLER_TYPE = 'simple'

DataRetrieverRequestHandler = __HANDLER_MAP.get(__USE_HANDLER_TYPE)

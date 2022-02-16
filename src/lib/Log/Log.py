import logging
import os
import sys

from lib.Config.ConfigUtil import ConfigUtil

_DEFAULT_FILE_NAME = 'unknown_module'
_DEFAULT_PATH_TO_WRITE = './'


class __Log:

    def __init__(self):
        module = sys.argv[0].split('/')[-1].rstrip('.py')
        try:
            logging_config = ConfigUtil.get_value_by_key('services', module, 'logging')
        except AttributeError as e:
            module = _DEFAULT_FILE_NAME
            logging_level = 'DEBUG'
            logging_path = _DEFAULT_PATH_TO_WRITE
        else:
            logging_level = logging_config.get('level')
            logging_path = logging_config.get('path')

        local_path = os.path.expanduser(logging_path)
        file_path = local_path + module + '.log'

        if not os.path.exists(local_path):
            os.makedirs(local_path)

        self.__logger = logging.getLogger(module)
        self.__logger.setLevel(logging.getLevelName(logging_level))

        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.getLevelName(logging_level))
        if logging_level == 'DEBUG':
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s [%(module)s.%(funcName)s]: %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p'))
        else:
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s: %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p'))
        self.__logger.addHandler(file_handler)

    def get_logger(self):
        return self.__logger


Logger = __Log().get_logger()

from abc import ABC, abstractmethod


class ConfigAbc(ABC):

    @abstractmethod
    def get_all_config(self):
        print('Inherited Class Needs to implement this function')

    @abstractmethod
    def get_value_by_key(self, *args):
        print('Inherited Class Needs to implement this function')

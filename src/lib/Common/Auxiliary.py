import math
# hard coding configurations for MangoDB
_MONGO_SERVER_IP = '10.10.10.111'
_MONGO_SERVER_PORT = 27017
_MONGO_DB_NAME = 'cnaf-sagw'


class __Auxiliary:

    __server_info = dict(ip=_MONGO_SERVER_IP, port=_MONGO_SERVER_PORT, db=_MONGO_DB_NAME)

    # Retrieve stream by ID
    def get_stream(self, streams, id):
        for stream in streams.values():
            if stream.get('stream_id') == id:
                return stream
        return None

    def get_mongo_config_server_info(self):

        return self.__server_info

    def generate_pagination_info(self, num_items: int, page_index: int, page_size: int):
        """

        :param num_items: number of items in the content list
        :param page_index: index of the page is requested
        :param page_size: the maximum number of items in each page
        :return: page_index: int, num_pages: int, begin_index: int, end_index: int
        """

        if num_items is None or num_items == 0:
            return 0, 0, 0, 0

        num_pages = math.ceil(num_items / page_size)

        if page_index >= num_pages:
            # get the last page
            page_index = num_pages
            begin_index = (page_index - 1) * page_size
            end_index = num_items
        else:
            begin_index = (page_index - 1) * page_size
            end_index = begin_index + page_size

        return page_index, num_pages, begin_index, end_index


Auxiliary = __Auxiliary()

from typing import Dict, List

import redis

from lib.Exceptions.CacheExceptions import PipelineUtilNotEnoughItemInCache, PipelineUtilNoSuchPipelineSetting


def _convert_to_redis_key(*args):
    key_segments = []
    for item in args:
        key_segments.append(str(item))
    redis_key = ':'.join(key_segments)
    return redis_key


CACHE_COLLECTION_KEY = 'collection_name'
CACHE_FIELD_KEY = 'fields'
CACHE_LEN_KEY = 'cache_len'


class CacheRedis:
    __client = None
    __settings: Dict[str, Dict[str, object]] = {}

    def __init__(self, host: str, port: int = 6379, pw: str = None):
        # Todo: redis.Redis support connection pool
        CacheRedis.__client = redis.Redis(host=host, port=port, password=pw, db=1)

    def set_pipeline_cache_setting(self, pipeline: str, setting: Dict[str, object]) -> None:
        CacheRedis.__settings.setdefault(pipeline, setting)

    def get_setting_by_pipeline(self, pipeline: str) -> Dict[str, object]:
        setting: Dict[str, object] = CacheRedis.__settings.get(pipeline)
        if setting is None:
            raise PipelineUtilNoSuchPipelineSetting('No pipeline[{}] existing in the settings'.format(pipeline))
        return setting

    def get_all_pipeline_settings(self) -> Dict[str, Dict[str, object]]:
        return CacheRedis.__settings

    def _check_key_exists(self, key: str) -> bool:
        return 0 != CacheRedis.__client.exists(key)

    def set_attribute(self, collection_name: str, unique_id: int, field: str, value: any, cache_len: int) -> None:
        redis_key: str = _convert_to_redis_key(collection_name, unique_id, field)
        CacheRedis.__client.rpush(redis_key, value)
        if CacheRedis.__client.llen(redis_key) > cache_len:
            CacheRedis.__client.ltrim(redis_key, 1, -1)

    def get_last_attribute(self, collection_name: str, unique_id: int, field: str) -> any:
        redis_key: str = _convert_to_redis_key(collection_name, unique_id, field)
        return CacheRedis.__client.lindex(redis_key, -1).decode('utf-8')

    def get_last_attribute_as_float(self, collection_name: str, unique_id: int, field: str) -> float:
        redis_key: str = _convert_to_redis_key(collection_name, unique_id, field)
        return float(CacheRedis.__client.lindex(redis_key, -1).decode('utf-8'))

    def get_all_attributes(self, collection_name: str, unique_id: int, field: str) -> List[any]:
        redis_key: str = _convert_to_redis_key(collection_name, unique_id, field)
        value: any = CacheRedis.__client.lrange(redis_key, 0, -1)
        return value

    def set_record(self, pipeline: str, unique_id: int, record: Dict[str, any]) -> None:
        pipeline_setting: Dict[str, any] = self.get_setting_by_pipeline(pipeline)
        for key in record:
            self.set_attribute(pipeline_setting.get(CACHE_COLLECTION_KEY), unique_id, key, record[key],
                               pipeline_setting.get(CACHE_LEN_KEY))

    def get_last_record(self, pipeline: str, type_id: int) -> Dict[str, any]:
        pipeline_setting: Dict[str, any] = self.get_setting_by_pipeline(pipeline)

        record = dict()
        for field in pipeline_setting.get(CACHE_FIELD_KEY):
            
            try:
                value = self.get_last_attribute(pipeline_setting.get(CACHE_COLLECTION_KEY), type_id, field)
                if value is None:
                    raise PipelineUtilNotEnoughItemInCache    
                float_val = float(value)
                value = float_val
            except ValueError:
                pass
            except AttributeError:
                raise PipelineUtilNotEnoughItemInCache

            record.setdefault(field, value)

        return record

    def get_all_records(self, pipeline: str, type_id: int) -> Dict[str, List[any]]:
        pipeline_setting: Dict[str, any] = self.get_setting_by_pipeline(pipeline)

        record = dict()
        for field in pipeline_setting.get(CACHE_FIELD_KEY):
            attributes = self.get_all_attributes(pipeline_setting.get(CACHE_COLLECTION_KEY), type_id,
                                                 field)
            if len(attributes) == 0:
                raise PipelineUtilNotEnoughItemInCache                                   
            # if len(attributes) < pipeline_setting.get(CACHE_LEN_KEY):
            #     raise PipelineUtilNotEnoughItemInCache(
            #         'Expect {} items, but only has {}'.format(pipeline_setting.get(CACHE_LEN_KEY),
            #                                                   len(attributes)))
            try:
                float_val = float(attributes[0])
                attributes = [float(item) for item in attributes]
            except ValueError:
                pass
            record.setdefault(field, attributes)

        return record

    def flushdb(self):
        return self.__client.flushdb()

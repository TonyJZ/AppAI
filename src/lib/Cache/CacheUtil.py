from typing import Dict, List

from lib.Cache.CacheRedis import CacheRedis, CACHE_COLLECTION_KEY, CACHE_FIELD_KEY, CACHE_LEN_KEY
from lib.Config.ConfigUtil import ConfigUtil


class _CacheHandler:
    __handler: CacheRedis = None

    def __init__(self):
        cache_config: Dict = ConfigUtil.get_value_by_key('resources', 'redis')
        _CacheHandler.__handler = CacheRedis(cache_config.get('redis_host_ip'))

        # load configuration and set up the settings
        stream_configs: Dict = ConfigUtil.get_value_by_key('streams')
        for key, value in stream_configs.items():
            pipelines = value.get('pipelines')
            for pipeline in pipelines:
                setting = {
                    CACHE_COLLECTION_KEY: pipeline.get('pipeline_name'),
                    CACHE_FIELD_KEY: pipeline.get('cache_fields'),
                    CACHE_LEN_KEY: pipeline.get('n_steps')
                }
                _CacheHandler.__handler.set_pipeline_cache_setting(pipeline.get('pipeline_name'), setting)

        print(_CacheHandler.__handler.get_all_pipeline_settings())

    def set_record(self, pipeline: str, type_id, record) -> None:
        _CacheHandler.__handler.set_record(pipeline, type_id, record)

    def get_last_record(self, pipeline, type_id) -> Dict[str, any]:
        return _CacheHandler.__handler.get_last_record(pipeline, type_id)

    def get_all_records(self, pipeline, type_id) -> Dict[str, List[any]]:
        return _CacheHandler.__handler.get_all_records(pipeline, type_id)

    def clear_all(self):
        _CacheHandler.__handler.flushdb()

# exporting only the CacheUtil from this module
CacheUtil = _CacheHandler()

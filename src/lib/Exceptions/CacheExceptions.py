from lib.Exceptions.CustomException import Error


class PipelineUtilNotEnoughItemInCache(Error):
    """Raised when the items in the cache are not enough for use"""
    pass


class PipelineUtilNoSuchPipelineSetting(Error):
    """Raised when there is no matching Pipeline setting in CacheUtil"""
    pass
from lib.Config.ConfigUtil import ConfigUtil
from lib.Cache.CacheUtil import CacheUtil
from lib.Storage.StorageUtil import StorageHandler
from lib.Storage.StorageUtil import __project_root_folder



# __project_root_folder = '/cnaf-sagw'
# load configuration
__hdfs_ip = ConfigUtil.get_value_by_key('resources', 'hdfs', 'hdfs_host_ip')

StorageUtil = StorageHandler(__hdfs_ip, __project_root_folder)
print(__hdfs_ip, __project_root_folder)

redis = True

raw = True
models = True
results = True


if raw:
    StorageUtil.delete_folder("{}/raw".format(__project_root_folder))

if models:
    StorageUtil.delete_folder("{}/models".format(__project_root_folder))

if results:
    StorageUtil.delete_folder("{}/results".format(__project_root_folder))

if redis:
    CacheUtil.clear_all()

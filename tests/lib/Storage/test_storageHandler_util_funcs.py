import unittest
from hdfs import InsecureClient, HdfsError

from lib.Storage.StorageUtil import StorageHandler

__hdfs_ip = '10.10.10.184'
__user = 'root'
__web_url = 'http://' + __hdfs_ip + ':50070'

_hdfs_root_folder = '/unittest'
_storage_util = StorageHandler(__hdfs_ip, _hdfs_root_folder)
_file_to_create = _hdfs_root_folder + '/test/test.txt'
_client = InsecureClient(__web_url, __user)


class TestStorageHandlerUtilFuncs(unittest.TestCase):

    def setUp(self) -> None:

        _client.write(_file_to_create, data='test')

    def tearDown(self) -> None:
        _client.delete(_hdfs_root_folder, recursive=True)

    def test_check_file_exists(self):
        split_by_slash = _file_to_create.split('/')
        path = '/'.join(split_by_slash[0:-1])
        file = split_by_slash[-1]
        rv = _storage_util.check_file_exists(path, file)
        self.assertTrue(rv)

    def test_delete_file(self):

        rv = _storage_util.delete_file(_file_to_create)
        self.assertTrue(rv)

    def test_delete_folder(self):
        rv = _storage_util.delete_folder(_hdfs_root_folder)
        self.assertTrue(rv)


if __name__ == '__main__':
    unittest.main()

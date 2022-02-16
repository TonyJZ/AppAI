import unittest

from tests.lib.Storage import test_storageHandler_read_funcs, test_storageHandler_util_funcs, test_storageHandler_write_funcs



def suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTest(loader.loadTestsFromModule(test_storageHandler_util_funcs))
    suite.addTest(loader.loadTestsFromModule(test_storageHandler_read_funcs))
    suite.addTest(loader.loadTestsFromModule(test_storageHandler_write_funcs))

    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(suite())
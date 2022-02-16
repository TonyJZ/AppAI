from lib.Exceptions.CustomException import Error


class NoModelInStorageException(Error):
    """Raised if there is no trained model in the storage for specific pipeline and type_id"""
    pass


class IncorrectStoragePathException(Error):
    """Raised if storage path is incorrect"""
    pass

class NoDataInStorageException(Error):
    """Raised if there is no data in the storage for specific stream and type_id"""
    pass
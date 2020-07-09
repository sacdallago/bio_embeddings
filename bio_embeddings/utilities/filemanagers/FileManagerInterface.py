import abc


class FileManagerInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def create_prefix(self, prefix) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def create_stage(self, prefix, stage) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def create_file(self, prefix, stage, file_name, extension=None) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def create_directory(self, prefix, stage, directory_name) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_file(self, prefix, stage, file_name, extension=None) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def exists(self, prefix, stage=None, file_name=None, extension=None) -> bool:
        raise NotImplementedError

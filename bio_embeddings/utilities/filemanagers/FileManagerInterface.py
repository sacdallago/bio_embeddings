import abc


class FileManagerInterface(object, metaclass=abc.ABCMeta):

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
    def get_file(self, prefix, stage, file_name) -> str:
        raise NotImplementedError

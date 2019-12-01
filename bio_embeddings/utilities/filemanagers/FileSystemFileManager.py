from bio_embeddings.utilities.filemanagers.FileManagerInterface import FileManagerInterface


class FileSystemFileManager(FileManagerInterface):

    def get_file(self, prefix, stage, file_name) -> str:
        pass

    def create_file(self, prefix, stage, file_name, extension=None) -> str:
        pass

    def create_stage(self, prefix, stage) -> str:
        pass

    def create_prefix(self, prefix) -> str:
        pass

import os
from pathlib import Path
from os import path as os_path
from bio_embeddings.utilities.filemanagers.FileManagerInterface import FileManagerInterface


class FileSystemFileManager(FileManagerInterface):

    def exists(self, prefix, stage=None, file_name=None, extension=None) -> bool:
        path = Path(prefix)

        if stage:
            path /= stage
        if file_name:
            path /= file_name + (extension if extension else "")

        return os_path.exists(path)

    def get_file(self, prefix, stage, file_name, extension=None) -> str:
        path = Path(prefix)

        if stage:
            path /= stage
        if file_name:
            path /= file_name + (extension if extension else "")

        return str(path)

    def create_file(self, prefix, stage, file_name, extension=None) -> str:
        path = Path(prefix)

        if stage:
            path /= stage

        path /= file_name + (extension if extension else "")

        try:
            with open(path, 'a'):
                os.utime(path, None)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)

        return str(path)

    def create_stage(self, prefix, stage) -> str:
        path = Path(prefix) / stage

        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)

        return str(path)

    def create_prefix(self, prefix) -> str:
        path = Path(prefix)

        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)

        return str(path)

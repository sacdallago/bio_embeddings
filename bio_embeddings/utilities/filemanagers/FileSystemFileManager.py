import os
from pathlib import Path
from os import path as os_path
from bio_embeddings.utilities.filemanagers.FileManagerInterface import FileManagerInterface
from bio_embeddings.utilities.logging import Logger


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
            with open(path, 'w'):
                os.utime(path, None)
        except OSError:
            Logger.warn("Failed to create file %s." % path)
        else:
            Logger.log("Successfully created the file %s." % path)

        return str(path)

    def create_directory(self, prefix, stage, directory_name) -> str:
        path = Path(prefix)

        if stage:
            path /= stage

        path /= directory_name

        try:
            os.mkdir(path)
        except OSError:
            Logger.warn("Failed to create directory %s." % path)
        else:
            Logger.log("Successfully created the directory %s." % path)

        return str(path)

    def create_stage(self, prefix, stage) -> str:
        path = Path(prefix) / stage

        try:
            os.mkdir(path)
        except OSError:
            Logger.warn("Failed to create stage directory %s." % path)
        else:
            Logger.log("Successfully created the stage directory %s." % path)

        return str(path)

    def create_prefix(self, prefix) -> str:
        path = Path(prefix)

        try:
            os.mkdir(path)
        except OSError:
            Logger.warn("Failed to create prefix directory %s." % path)
        else:
            Logger.log("Successfully created the prefix directory %s." % path)

        return str(path)

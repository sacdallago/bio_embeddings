import logging
import os
from os import path as os_path
from pathlib import Path

from bio_embeddings.utilities.filemanagers.FileManagerInterface import FileManagerInterface

logger = logging.getLogger(__name__)


class FileSystemFileManager(FileManagerInterface):

    def __init__(self):
        super().__init__()

    def exists(self, prefix, stage=None, file_name=None, extension=None) -> bool:
        path = Path(prefix)

        if stage:
            path /= stage
        if file_name:
            path /= file_name + (extension or "")

        return os_path.exists(path)

    def get_file(self, prefix, stage, file_name, extension=None) -> str:
        path = Path(prefix)

        if stage:
            path /= stage
        if file_name:
            path /= file_name + (extension or "")

        return str(path)

    def create_file(self, prefix, stage, file_name, extension=None) -> str:
        path = Path(prefix)

        if stage:
            path /= stage

        path /= file_name + (extension or "")

        try:
            with open(path, 'w'):
                os.utime(path, None)
        except OSError as e:
            logger.error("Failed to create file %s" % path)
            raise e
        else:
            logger.info("Created the file %s" % path)

        return str(path)

    def create_directory(self, prefix, stage, directory_name) -> str:
        path = Path(prefix)

        if stage:
            path /= stage

        path /= directory_name

        try:
            os.mkdir(path)
        except FileExistsError:
            logger.info("Directory %s already exists." % path)
        except OSError as e:
            logger.error("Failed to create directory %s" % path)
            raise e
        else:
            logger.info("Created the directory %s" % path)

        return str(path)

    def create_stage(self, prefix, stage) -> str:
        path = Path(prefix) / stage

        try:
            os.mkdir(path)
        except FileExistsError:
            logger.info("Stage directory %s already exists." % path)
        except OSError as e:
            logger.error("Failed to create stage directory %s" % path)
            raise e
        else:
            logger.info("Created the stage directory %s" % path)

        return str(path)

    def create_prefix(self, prefix) -> str:
        path = Path(prefix)

        try:
            os.mkdir(path)
        except FileExistsError:
            logger.info("Prefix directory %s already exists." % path)
        except OSError as e:
            logger.error("Failed to create prefix directory %s" % path)
            raise e
        else:
            logger.info("Created the prefix directory %s" % path)

        return str(path)

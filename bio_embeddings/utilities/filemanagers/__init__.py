from bio_embeddings.utilities.filemanagers.FileSystemFileManager import FileSystemFileManager
from bio_embeddings.utilities.filemanagers.FileManagerInterface import FileManagerInterface


def get_file_manager(**kwargs):

    # A useless call to pacify the linters
    # TODO: when new FileManagers are available, parse the file manager type from "management".
    kwargs.get('management', {})

    return FileSystemFileManager()

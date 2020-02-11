from bio_embeddings.utilities.filemanagers.FileSystemFileManager import FileSystemFileManager
from bio_embeddings.utilities.filemanagers.FileManagerInterface import FileManagerInterface

FILE_MANAGERS = {
    'filesystem': FileSystemFileManager,
    None: FileSystemFileManager
}


def get_file_manager(**kwargs):
    management = kwargs.get('management', {})
    file_manager_type = management.get('file_manager')
    file_manager = FILE_MANAGERS.get(file_manager_type)

    return file_manager(**management)

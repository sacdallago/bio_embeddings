from bio_embeddings.utilities.filemanagers import FileSystemFileManager, FileManagerInterface

FILE_MANAGERS = {
    'filesystem': FileSystemFileManager,
    None: FileSystemFileManager
}


def get_file_manager(**kwargs) -> FileManagerInterface:
    file_manager = kwargs.get('management', {}).get('file_manager')

    return FILE_MANAGERS.get(file_manager)

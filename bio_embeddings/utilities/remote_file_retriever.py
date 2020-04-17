import os
import tempfile
import zipfile
from pathlib import Path
from urllib import request
from tqdm import tqdm
from bio_embeddings.utilities.exceptions import MissingParameterError, CannotFindDefaultFile, FileDoesntExistError
from bio_embeddings.utilities.config import read_config_file
from bio_embeddings.utilities.logging import Logger

_module_dir = Path(os.path.dirname(os.path.abspath(__file__)))
_defaults = read_config_file(_module_dir / 'defaults.yml')


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_model_directories_from_zip(model=None, directory=None, path=None) -> None:
    if not path:
        raise MissingParameterError("Missing required parameter: 'path'")

    url = _defaults.get(model, {}).get(directory)

    if not url:
        raise CannotFindDefaultFile("Trying to get file '{}' for model '{}', but doesn't exist.".format("model_folder_zip", path))

    if not os.path.exists(path):
        os.makedirs(path)

    f = tempfile.NamedTemporaryFile()

    file_name = f.name

    Logger.log("Downloading '{}' for model '{}' and storing in '{}'.".format("model_folder_zip", model, file_name))

    with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        request.urlretrieve(url, filename=file_name, reporthook=t.update_to)

    Logger.log("Unzipping '{}' for model '{}' and storing in '{}'.".format(f, model, path))

    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(path)


def get_model_file(model=None, file=None, path=None) -> None:
    if not path:
        raise MissingParameterError("Missing required parameter: 'path'")

    url = _defaults.get(model, {}).get(file)

    if not url:
        raise CannotFindDefaultFile("Trying to get file '{}' for model '{}', but doesn't exist.".format(file, path))

    if not os.path.isfile(path):
        raise FileDoesntExistError("Trying to open file, but doesn't exist.".format(path))

    Logger.log("Downloading '{}' for model '{}' and storing in '{}'.".format(file, model, path))

    with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        request.urlretrieve(url, filename=path, reporthook=t.update_to)


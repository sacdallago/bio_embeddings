import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional
from urllib import request

from appdirs import user_cache_dir
from tqdm import tqdm

from bio_embeddings.utilities.config import read_config_file

_module_dir: Path = Path(os.path.dirname(os.path.abspath(__file__)))
_defaults: Dict[str, Dict[str, str]] = read_config_file(_module_dir / "defaults.yml")

logger = logging.getLogger(__name__)


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


def get_model_directories_from_zip(
    model: Optional[str] = None,
    directory: Optional[str] = None,
    overwrite_cache: bool = False,
) -> str:
    """If the specified asset directory for the model is in the user cache, returns the
    directory path, otherwise downloads the zipped directory, unpacks in the cache and
    returns the location"""
    cache_path = (
        Path(user_cache_dir("bio_embeddings")).joinpath(model).joinpath(directory)
    )
    if (
        not overwrite_cache
        and cache_path.is_dir()
        and len(list(cache_path.iterdir())) > 1
    ):
        logger.info(f"Loading {directory} for {model} from cache at '{cache_path}'")
        return str(cache_path)

    cache_path.mkdir(parents=True, exist_ok=True)
    url = _defaults.get(model, {}).get(directory)

    # Since the directory are not user provided, this must never happen
    assert url, f"Directory {directory} for {model} doesn't exist."

    with tempfile.NamedTemporaryFile() as f:
        file_name = f.name

        logger.info(
            "Downloading {} for {} and storing in '{}'.".format(
                "model_folder_zip", model, file_name
            )
        )

        with TqdmUpTo(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            request.urlretrieve(url, filename=file_name, reporthook=t.update_to)

        logger.info(
            "Unzipping {} for {} and storing in '{}'.".format(
                file_name, model, cache_path
            )
        )

        with zipfile.ZipFile(file_name, "r") as zip_ref:
            zip_ref.extractall(cache_path)

    return str(cache_path)


def get_model_file(
    model: Optional[str] = None,
    file: Optional[str] = None,
    overwrite_cache: bool = False,
) -> str:
    """If the specified asset for the model is in the user cache, returns the
    location, otherwise downloads the file to cache and returns the location"""
    cache_path = Path(user_cache_dir("bio_embeddings")).joinpath(model).joinpath(file)
    if not overwrite_cache and cache_path.is_file():
        logger.info(f"Loading {file} for {model} from cache at '{cache_path}'")
        return str(cache_path)

    cache_path.parent.mkdir(exist_ok=True, parents=True)
    url = _defaults.get(model, {}).get(file)

    # Since the files are not user provided, this must never happen
    assert url, f"File {file} for {model} doesn't exist."

    logger.info(f"Downloading {file} for {model} and storing it in '{cache_path}'")

    with TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as t:
        request.urlretrieve(url, filename=cache_path, reporthook=t.update_to)

    return str(cache_path)

from urllib import request
from bio_embeddings.utilities import MissingParameterError, CannotFindDefaultFile
from bio_embeddings.utilities.config import read_config_file
from bio_embeddings.utilities.logging import Logger

# TODO: get abs. path for defaults.yml
defaults = read_config_file('defaults.yml')


def get_model_file(model=None, file=None, path=None) -> None:
    if not path:
        raise MissingParameterError("Missing required parameter: 'path'")

    url = defaults.get(model, {}).get(file, {})

    if not url:
        raise CannotFindDefaultFile("Trying to get file '{}' for model '{}', but doesn't exist.".format(file, path))

    Logger.log("Downloading '{}' for model '{}' and storing in '{}'.".format(file, model, path))

    request.urlretrieve(url, path)


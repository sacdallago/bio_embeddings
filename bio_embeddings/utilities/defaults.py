import tempfile
from bio_embeddings.utilities.logging import Logger
from urllib import request

ELMO_V1_WEIGHTS = "http://maintenance.dallago.us/public/embeddings/elmo/weights"
ELMO_V1_OPTIONS = "http://maintenance.dallago.us/public/embeddings/elmo/options"


def _get_elmo_v1():
    """

    :return: weight_file, options_file
    """

    Logger.log("Downloading weights and options file for ELMO v1 embedder")

    weight_file = tempfile.NamedTemporaryFile()
    options_file = tempfile.NamedTemporaryFile()

    request.urlretrieve(ELMO_V1_WEIGHTS, weight_file.name)
    request.urlretrieve(ELMO_V1_OPTIONS, options_file.name)

    Logger.log("Downloaded weights and options file for ELMO v1 embedder")

    return weight_file, options_file


_EMBEDDERS = {
    "elmov1": _get_elmo_v1,
    None: lambda x: Logger.log("Trying to get undefined embedder. Name: {}".format(x))
}


def get_defaults(embedder):
    return _EMBEDDERS.get(embedder)()

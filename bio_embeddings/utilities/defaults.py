import tempfile
from bio_embeddings.utilities.logging import Logger
from urllib import request

ELMO_V1_WEIGHTS = "http://maintenance.dallago.us/public/embeddings/elmo/weights"
ELMO_V1_OPTIONS = "http://maintenance.dallago.us/public/embeddings/elmo/options"
ELMO_V1_SUBCELLULAR_LOCATION_CHECKPOINT = "http://maintenance.dallago.us/public/embeddings/elmo/subcell"
ELMO_V1_SECONDARY_STRUCTURE_CHECKPOINT = "http://maintenance.dallago.us/public/embeddings/elmo/secstruct"


def _get_elmo_v1():
    """

    :return: weight_file, options_file
    """

    Logger.log("Downloading files ELMO v1 embedder")

    weight_file = tempfile.NamedTemporaryFile()
    options_file = tempfile.NamedTemporaryFile()
    subcellular_location_checkpoint = tempfile.NamedTemporaryFile()
    secondary_structure_checkpoint_file = tempfile.NamedTemporaryFile()

    request.urlretrieve(ELMO_V1_WEIGHTS, weight_file.name)
    request.urlretrieve(ELMO_V1_OPTIONS, options_file.name)
    request.urlretrieve(ELMO_V1_SUBCELLULAR_LOCATION_CHECKPOINT, subcellular_location_checkpoint.name)
    request.urlretrieve(ELMO_V1_SECONDARY_STRUCTURE_CHECKPOINT, secondary_structure_checkpoint_file.name)

    Logger.log("Downloaded files for ELMO v1 embedder")

    return weight_file, options_file, subcellular_location_checkpoint, secondary_structure_checkpoint_file


_EMBEDDERS = {
    "elmov1": _get_elmo_v1,
    None: lambda x: Logger.log("Trying to get undefined embedder. Name: {}".format(x))
}


def get_defaults(embedder):
    return _EMBEDDERS.get(embedder)()

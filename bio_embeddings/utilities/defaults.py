import tempfile
from bio_embeddings.utilities.logging import Logger
from urllib import request

ELMO_V1_WEIGHTS = "http://maintenance.dallago.us/public/embeddings/embedding_models/seqvec/weights.hdf5"
ELMO_V1_OPTIONS = "http://maintenance.dallago.us/public/embeddings/embedding_models/seqvec/options.json"
ELMO_V1_SUBCELLULAR_LOCATION_CHECKPOINT = "http://maintenance.dallago.us/public/embeddings/feature_models/seqvec/subcell_checkpoint.pt"
ELMO_V1_SECONDARY_STRUCTURE_CHECKPOINT = "http://maintenance.dallago.us/public/embeddings/feature_models/seqvec/secstruct_checkpoint.pt"


def _get_elmo_v1():
    """

    :return: weight_file, options_file, subcellular_location_checkpoint, secondary_structure_checkpoint_file
    """

    Logger.log("Downloading files ELMO v1 embedder")

    weight_file = tempfile.NamedTemporaryFile()
    options_file = tempfile.NamedTemporaryFile()
    subcellular_location_checkpoint = tempfile.NamedTemporaryFile()
    secondary_structure_checkpoint_file = tempfile.NamedTemporaryFile()

    Logger.log("Downloading weights from {}".format(ELMO_V1_WEIGHTS))
    request.urlretrieve(ELMO_V1_WEIGHTS, weight_file.name)
    Logger.log("Downloading options from {}".format(ELMO_V1_OPTIONS))
    request.urlretrieve(ELMO_V1_OPTIONS, options_file.name)
    Logger.log("Downloading subcellular location checkpoint from {}".format(ELMO_V1_SUBCELLULAR_LOCATION_CHECKPOINT))
    request.urlretrieve(ELMO_V1_SUBCELLULAR_LOCATION_CHECKPOINT, subcellular_location_checkpoint.name)
    Logger.log("Downloading secondary structure checkpoint from {}".format(ELMO_V1_SECONDARY_STRUCTURE_CHECKPOINT))
    request.urlretrieve(ELMO_V1_SECONDARY_STRUCTURE_CHECKPOINT, secondary_structure_checkpoint_file.name)

    Logger.log("Downloaded files for ELMO v1 embedder")

    return weight_file, options_file, subcellular_location_checkpoint, secondary_structure_checkpoint_file


WORD2VEC_MODEL = "http://maintenance.dallago.us/public/embeddings/embedding_models/word2vec/word2vec.model"


def _get_word2vec():
    Logger.log("Downloading files word2vec embedder")

    model_file = tempfile.NamedTemporaryFile()

    Logger.log("Downloading model file from {}".format(WORD2VEC_MODEL))
    request.urlretrieve(WORD2VEC_MODEL, model_file.name)

    Logger.log("Downloaded files for word2vec embedder")

    return model_file


FASTTEXT_MODEL = "http://maintenance.dallago.us/public/embeddings/embedding_models/fasttext/fasttext.model"


def _get_fasttext():
    Logger.log("Downloading files fasttext embedder")

    model_file = tempfile.NamedTemporaryFile()

    Logger.log("Downloading model file from {}".format(FASTTEXT_MODEL))
    request.urlretrieve(FASTTEXT_MODEL, model_file.name)

    Logger.log("Downloaded files for fasttext embedder")

    return model_file


_EMBEDDERS = {
    "elmov1": _get_elmo_v1,
    "word2vec": _get_word2vec,
    "fasttext": _get_fasttext,
    None: lambda x: Logger.log("Trying to get undefined embedder. Name: {}".format(x))
}


def get_defaults(embedder):
    return _EMBEDDERS.get(embedder)()

import tempfile
from bio_embeddings.utilities.logging import Logger
from urllib import request

SEQVEC_V1_WEIGHTS = "http://maintenance.dallago.us/public/embeddings/embedding_models/seqvec/weights.hdf5"
SEQVEC_V1_OPTIONS = "http://maintenance.dallago.us/public/embeddings/embedding_models/seqvec/options.json"


def _get_seqvec_v1():
    """
    :return: weight_file, options_file
    """

    Logger.log("Downloading files ELMO v1 embedder")

    weight_file = tempfile.NamedTemporaryFile()
    options_file = tempfile.NamedTemporaryFile()

    Logger.log("Downloading weights from {}".format(SEQVEC_V1_WEIGHTS))
    request.urlretrieve(SEQVEC_V1_WEIGHTS, weight_file.name)
    Logger.log("Downloading options from {}".format(SEQVEC_V1_OPTIONS))
    request.urlretrieve(SEQVEC_V1_OPTIONS, options_file.name)

    Logger.log("Downloaded files for ELMO v1 embedder")

    return weight_file, options_file


SEQVEC_V1_SUBCELLULAR_LOCATION_CHECKPOINT = "http://maintenance.dallago.us/public/embeddings/feature_models/seqvec/subcell_checkpoint.pt"
SEQVEC_V1_SECONDARY_STRUCTURE_CHECKPOINT = "http://maintenance.dallago.us/public/embeddings/feature_models/seqvec/secstruct_checkpoint.pt"


def _get_seqvec_feature_extractors_v1():
    """

    :return: subcellular_location_checkpoint, secondary_structure_checkpoint_file
    """

    Logger.log("Downloading files SEQVEC v1 feature extractors")

    subcellular_location_checkpoint = tempfile.NamedTemporaryFile()
    secondary_structure_checkpoint_file = tempfile.NamedTemporaryFile()

    Logger.log("Downloading subcellular location checkpoint from {}".format(SEQVEC_V1_SUBCELLULAR_LOCATION_CHECKPOINT))
    request.urlretrieve(SEQVEC_V1_SUBCELLULAR_LOCATION_CHECKPOINT, subcellular_location_checkpoint.name)
    Logger.log("Downloading secondary structure checkpoint from {}".format(SEQVEC_V1_SECONDARY_STRUCTURE_CHECKPOINT))
    request.urlretrieve(SEQVEC_V1_SECONDARY_STRUCTURE_CHECKPOINT, secondary_structure_checkpoint_file.name)

    Logger.log("Downloaded files for ELMO v1 feature extractors")

    return subcellular_location_checkpoint, secondary_structure_checkpoint_file


SEQVEC_V2_WEIGHTS = "http://maintenance.dallago.us/public/embeddings/embedding_models/seqvec_v2/weights.hdf5"
SEQVEC_V2_OPTIONS = "http://maintenance.dallago.us/public/embeddings/embedding_models/seqvec_v2/options.json"
SEQVEC_V2_VOCABULARY = "http://maintenance.dallago.us/public/embeddings/embedding_models/seqvec_v2/vocab.txt"


def _get_seqvec_v2():
    """

    :return: weight_file, options_file, subcellular_location_checkpoint, secondary_structure_checkpoint_file
    """

    Logger.log("Downloading files ELMO v2 embedder")

    weight_file = tempfile.NamedTemporaryFile()
    options_file = tempfile.NamedTemporaryFile()
    vocabulary_file = tempfile.NamedTemporaryFile()

    Logger.log("Downloading weights from {}".format(SEQVEC_V2_WEIGHTS))
    request.urlretrieve(SEQVEC_V2_WEIGHTS, weight_file.name)
    Logger.log("Downloading options from {}".format(SEQVEC_V2_OPTIONS))
    request.urlretrieve(SEQVEC_V2_OPTIONS, options_file.name)
    Logger.log("Downloading vocabulary from {}".format(SEQVEC_V2_VOCABULARY))
    request.urlretrieve(SEQVEC_V2_VOCABULARY, vocabulary_file.name)

    Logger.log("Downloaded files for ELMO v2 embedder")

    return weight_file, options_file, vocabulary_file


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


GLOVE_MODEL = "http://maintenance.dallago.us/public/embeddings/embedding_models/glove/glove.model"


def _get_glove():
    Logger.log("Downloading files glove embedder")

    model_file = tempfile.NamedTemporaryFile()

    Logger.log("Downloading model file from {}".format(GLOVE_MODEL))
    request.urlretrieve(GLOVE_MODEL, model_file.name)

    Logger.log("Downloaded files for glove embedder")

    return model_file


TRANSFORMER_BASE_MODEL = "http://maintenance.dallago.us/public/embeddings/embedding_models/transformerxl_base/model.pt"
TRANSFORMER_BASE_VOCABULARY = "http://maintenance.dallago.us/public/embeddings/embedding_models/transformerxl_base/vocab.pt"


def _get_transformer_base():
    Logger.log("Downloading files transformer_base embedder")

    model_file = tempfile.NamedTemporaryFile()
    vocabulary_file = tempfile.NamedTemporaryFile()

    Logger.log("Downloading model file from {}".format(TRANSFORMER_BASE_MODEL))
    request.urlretrieve(TRANSFORMER_BASE_MODEL, model_file.name)

    Logger.log("Downloading vocabulary file from {}".format(TRANSFORMER_BASE_VOCABULARY))
    request.urlretrieve(TRANSFORMER_BASE_VOCABULARY, vocabulary_file.name)

    Logger.log("Downloaded files for transformer_base embedder")

    return model_file, vocabulary_file


# TODO: Implement!
def _get_transformer_large():
    return _get_transformer_base()


_EMBEDDERS = {
    "seqvecv1": _get_seqvec_v1,
    "seqvecv1_feature_extractors": _get_seqvec_feature_extractors_v1,
    "seqvecv2": _get_seqvec_v2,
    "word2vec": _get_word2vec,
    "fasttext": _get_fasttext,
    "glove": _get_glove,
    "transformer_base": _get_transformer_base,
    "transformer_large": _get_transformer_large,
    None: lambda x: Logger.log("Trying to get undefined embedder. Name: {}".format(x))
}


def get_model_parameters(embedder):
    return _EMBEDDERS.get(embedder)()

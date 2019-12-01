import numpy as np
from collections import defaultdict
from copy import deepcopy
from bio_embeddings.embed.seqvec import SeqVecEmbedder
from bio_embeddings.utilities import InvalidParameterError, get_model_file
from bio_embeddings.utilities.filemanagers import get_file_manager
from bio_embeddings.utilities.helpers import check_required

_STAGE_NAME = "embed"
_FILE_MANAGER = None


def seqvec(**kwargs):
    necessary_files = ['weights_file', 'options_file']
    result_kwargs = deepcopy(kwargs)

    if result_kwargs.get('seqvec_version') == 2 or result_kwargs.get('vocabulary_file'):
        necessary_files.append('vocabulary_file')
        result_kwargs['seqvec_version'] = 2

    for file in necessary_files:
        if not result_kwargs.get(file):
            file_path = _FILE_MANAGER.create_file(kwargs.get('prefix'), _STAGE_NAME, file)

            get_model_file(
                model='seqvecv{}'.format(str(result_kwargs['seqvec_version'])),
                file= file,
                path=file_path
            )

            result_kwargs[file] = file_path


    # TODO: extract sequences from fasta file
    sequences = read_fasta(kwargs['sequences_file'])
    embedder = SeqVecEmbedder(**kwargs)

    # TODO: get the embeddings for many sequences
    embeddings = embedder.embed_many(sequences)

    # TODO: save embedding to file
    embeddings_file_path = _FILE_MANAGER.create_file(kwargs.get('prefix'), _STAGE_NAME, 'embeddings_file', extension='.npy')
    np.save(embeddings_file_path, embeddings)
    result_kwargs['embeddings_file'] = embeddings_file_path

    return result_kwargs


def fasttext(**kwargs):
    pass


def glove(**kwargs):
    pass


def transformerxl(**kwargs):
    pass


def word2vec(**kwargs):
    pass


# list of available embedding protocols
PROTOCOLS = {
    "seqvec": seqvec,
    "fasttext": fasttext,
    "glove": glove,
    "transformerxl": transformerxl,
    "word2vec": word2vec
}


def run(**kwargs):
    """
    Run embedding protocol

    Parameters
    ----------
    kwargs arguments (* denotes optional):
        sequences_file: Where sequences live
        prefix: Output prefix for all generated files
        protocol: Which embedder to use

    Returns
    -------
    Dictionary with results of stage in following fields (in brackets - not returned by all protocols):

        * sequences_file
        * prefix
        * protocol
        * embeddings_file
        * [seqvec_version]
        * [weights_file]
        * [option_file]
        * [vocabulary_file]
        * [model_file]
    """
    check_required(kwargs, ["protocol"])

    if kwargs["protocol"] not in PROTOCOLS:
        raise InvalidParameterError(
            "Invalid protocol selection: " +
            "{}. Valid protocols are: {}".format(
                kwargs["protocol"], ", ".join(PROTOCOLS.keys())
            )
        )

    _FILE_MANAGER = get_file_manager(**kwargs)

    return PROTOCOLS[kwargs["protocol"]](**kwargs)
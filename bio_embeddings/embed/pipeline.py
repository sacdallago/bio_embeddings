import h5py
from copy import deepcopy
from bio_embeddings.embed.seqvec import SeqVecEmbedder
from bio_embeddings.utilities import InvalidParameterError, get_model_file, \
    check_required, get_file_manager, read_fasta_file


def seqvec(**kwargs):
    necessary_files = ['weights_file', 'options_file']
    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager(**kwargs)

    if result_kwargs.get('seqvec_version') == 2 or result_kwargs.get('vocabulary_file'):
        necessary_files.append('vocabulary_file')
        result_kwargs['seqvec_version'] = 2

    for file in necessary_files:
        if not result_kwargs.get(file):
            file_path = file_manager.create_file(result_kwargs.get('prefix'), result_kwargs.get('stage_name'), file)

            get_model_file(
                model='seqvecv{}'.format(str(result_kwargs.get('seqvec_version', 1))),
                file=file,
                path=file_path
            )

            result_kwargs[file] = file_path

    proteins = read_fasta_file(result_kwargs['remapped_sequences_file'])
    embedder = SeqVecEmbedder(**result_kwargs)

    embeddings = embedder.embed_many([protein.seq for protein in proteins])

    embeddings_file_path = file_manager.create_file(result_kwargs.get('prefix'), result_kwargs.get('stage_name'), 'embeddings_file', extension='.h5')
    with h5py.File(embeddings_file_path, "w") as hf:
        for i, protein in enumerate(proteins):
            hf.create_dataset(protein.id, data=embeddings[i])
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

    return PROTOCOLS[kwargs["protocol"]](**kwargs)

import h5py
from copy import deepcopy
from bio_embeddings.embed.seqvec import SeqVecEmbedder
from bio_embeddings.embed.albert import AlbertEmbedder
from bio_embeddings.utilities import InvalidParameterError, get_model_file, \
    check_required, get_file_manager, read_fasta_file, Logger, get_model_directories_from_zip


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

    # Create embeddings file
    embeddings_file_path = file_manager.create_file(result_kwargs.get('prefix'), result_kwargs.get('stage_name'),
                                                    'embeddings_file', extension='.h5')
    embeddings_file = h5py.File(embeddings_file_path, "w")
    result_kwargs['embeddings_file'] = embeddings_file_path

    reduced_embeddings_file = None
    if result_kwargs.get('reduce') is True:
        reduced_embeddings_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                result_kwargs.get('stage_name'),
                                                                'reduced_embeddings_file', extension='.h5')
        result_kwargs['reduced_embeddings_file'] = reduced_embeddings_file_path
        reduced_embeddings_file = h5py.File(reduced_embeddings_file_path, "w")

    # Get embedder
    embedder = SeqVecEmbedder(**result_kwargs)

    # Embed iteratively (5k sequences at the time)
    chunk_size = 5000
    warning_size = 15000

    for i in range(0, len(proteins), chunk_size):
        if any(y > warning_size for y in [len(prot) for prot in proteins[i:i+chunk_size]]):
            Logger.warn(
                "Generating SeqVec embeddings for proteins with length > {} "
                "is slow and may fail due to memory consumption!".format(warning_size)
            )

        embeddings = embedder.embed_many([protein.seq for protein in proteins[i:i+chunk_size]])

        for index, protein in enumerate(proteins[i:i+chunk_size]):
            embeddings_file.create_dataset(protein.id, data=embeddings[index])
            if result_kwargs.get('reduce') is True:
                reduced_embeddings_file.create_dataset(protein.id, data=SeqVecEmbedder.reduce_per_protein(embeddings[index]))

    embeddings_file.close()
    if result_kwargs.get('reduce') is True:
        reduced_embeddings_file.close()

    return result_kwargs


def albert(**kwargs):
    necessary_directories = ['model_directory']
    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager(**kwargs)

    for directory in necessary_directories:
        if not result_kwargs.get(directory):
            directory_path = file_manager.create_directory(result_kwargs.get('prefix'), result_kwargs.get('stage_name'), directory)

            get_model_directories_from_zip(
                model='albert',
                directory=directory,
                path=directory_path
            )

            result_kwargs[directory] = directory_path

    proteins = read_fasta_file(result_kwargs['remapped_sequences_file'])

    # Create embeddings file
    embeddings_file_path = file_manager.create_file(result_kwargs.get('prefix'), result_kwargs.get('stage_name'),
                                                    'embeddings_file', extension='.h5')
    embeddings_file = h5py.File(embeddings_file_path, "w")
    result_kwargs['embeddings_file'] = embeddings_file_path

    reduced_embeddings_file = None
    if result_kwargs.get('reduce') is True:
        reduced_embeddings_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                result_kwargs.get('stage_name'),
                                                                'reduced_embeddings_file', extension='.h5')
        result_kwargs['reduced_embeddings_file'] = reduced_embeddings_file_path
        reduced_embeddings_file = h5py.File(reduced_embeddings_file_path, "w")

    # Get embedder
    embedder = AlbertEmbedder(**result_kwargs)

    # Embed iteratively (5k sequences at the time)
    chunk_size = 5000
    warn_size = 510

    for i in range(0, len(proteins), chunk_size):
        if any(y > warn_size for y in [len(prot) for prot in proteins[i:i + chunk_size]]):
            Logger.warn(
                "Batch contains proteins longer than {}AA. "
                "The embeddings for these proteins will be empty.".format(warn_size)
            )

        embeddings = embedder.embed_many([protein.seq for protein in proteins[i:i + chunk_size]])

        for index, protein in enumerate(proteins[i:i + chunk_size]):
            embeddings_file.create_dataset(protein.id, data=embeddings[index])
            if result_kwargs.get('reduce') is True:
                reduced_embeddings_file.create_dataset(protein.id,
                                                       data=AlbertEmbedder.reduce_per_protein(embeddings[index]))

    embeddings_file.close()
    if result_kwargs.get('reduce') is True:
        reduced_embeddings_file.close()

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
    "word2vec": word2vec,
    "albert": albert
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

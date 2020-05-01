import h5py
from pandas import read_csv
from tqdm import tqdm
from copy import deepcopy
from bio_embeddings.embed.seqvec import SeqVecEmbedder
from bio_embeddings.embed.albert import AlbertEmbedder
from bio_embeddings.utilities import InvalidParameterError, get_model_file, \
    check_required, get_file_manager, Logger, get_model_directories_from_zip, read_fasta_file_generator


def seqvec(**kwargs):
    necessary_files = ['weights_file', 'options_file']
    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager(**kwargs)

    # Initialize pipeline and model specific options:
    result_kwargs['max_amino_acids'] = result_kwargs.get("max_amino_acids", 15000)
    result_kwargs['max_amino_acids_RAM'] = result_kwargs.get("max_amino_acids_RAM", 100000)

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

    # Create reduced embeddings file if set in params
    result_kwargs['reduce'] = result_kwargs.get('reduce', False)

    reduced_embeddings_file = None
    if result_kwargs['reduce'] is True:
        reduced_embeddings_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                result_kwargs.get('stage_name'),
                                                                'reduced_embeddings_file', extension='.h5')
        result_kwargs['reduced_embeddings_file'] = reduced_embeddings_file_path
        reduced_embeddings_file = h5py.File(reduced_embeddings_file_path, "w")

    # Create embeddings file if not discarted in params
    embeddings_file = None

    result_kwargs['discard_per_amino_acid_embeddings'] = result_kwargs.get('discard_per_amino_acid_embeddings', False)

    if result_kwargs['discard_per_amino_acid_embeddings'] is True:
        if result_kwargs['reduce'] is False:
            raise InvalidParameterError("Cannot have discard_per_amino_acid_embeddings=True and reduce=False. Both must be True.")
    else:
        embeddings_file_path = file_manager.create_file(result_kwargs.get('prefix'), result_kwargs.get('stage_name'),
                                                        'embeddings_file', extension='.h5')
        embeddings_file = h5py.File(embeddings_file_path, "w")
        result_kwargs['embeddings_file'] = embeddings_file_path

    # Get embedder
    embedder = SeqVecEmbedder(**result_kwargs)

    # Embed iteratively (5k sequences at the time)
    max_amino_acids_RAM = result_kwargs['max_amino_acids_RAM']
    protein_generator = read_fasta_file_generator(result_kwargs['remapped_sequences_file'])

    # Get sequence mapping to use as information source
    mapping_file = read_csv(result_kwargs['mapping_file'], index_col=0)

    candidates = list()
    aa_count = 0

    for _, sequence in tqdm(enumerate(protein_generator), total=len(mapping_file)):
        candidates.append(sequence)
        aa_count += len(sequence)

        # If a single sequence has more AA than allowed in max_amino_acids, switch to CPU
        if len(sequence) > result_kwargs['max_amino_acids']:
            Logger.warn(
                '''One sequence in your set has length {}, which is more than what is defined in the max_amino_acids parameter ({}).
                
                To avoid running out of GPU memory, the pipeline will now use the CPU instead of the GPU to calculate embeddings.
                This allows to embed much longer sequences (since using main RAM instead of GPU RAM), but comes at a significant speed deacreas (CPU instead of GPU computing).
                
                If you think your GPU RAM can handle longer sequences, try increasing max_amino_acids.
                As a rule of thumb: ~15000 AA require 5.5GB of GPU RAM and can be embedded on a GTX1080 with 8GB.'''.format(len(sequence), result_kwargs['max_amino_acids']))

            result_kwargs['use_cpu'] = True
            embedder = SeqVecEmbedder(**result_kwargs)

        if aa_count + len(sequence) > max_amino_acids_RAM:
            embeddings = embedder.embed_many([protein.seq for protein in candidates])

            for index, protein in enumerate(candidates):
                if result_kwargs.get('discard_per_amino_acid_embeddings') is False:
                    embeddings_file.create_dataset(protein.id, data=embeddings[index])

                if result_kwargs.get('reduce') is True:
                    reduced_embeddings_file.create_dataset(
                        protein.id,
                        data=embedder.reduce_per_protein(embeddings[index])
                    )

            # Reset
            aa_count = 0
            candidates = list()

    if candidates:
        embeddings = embedder.embed_many([protein.seq for protein in candidates])

        for index, protein in enumerate(candidates):
            if result_kwargs.get('discard_per_amino_acid_embeddings') is False:
                embeddings_file.create_dataset(protein.id, data=embeddings[index])

            if result_kwargs.get('reduce') is True:
                reduced_embeddings_file.create_dataset(
                    protein.id,
                    data=embedder.reduce_per_protein(embeddings[index])
                )

    # Close embeddings files
    if result_kwargs.get('discard_per_amino_acid_embeddings') is False:
        embeddings_file.close()

    if result_kwargs.get('reduce') is True:
        reduced_embeddings_file.close()

    return result_kwargs


def albert(**kwargs):
    necessary_directories = ['model_directory']
    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager(**kwargs)

    # Initialize pipeline and model specific options:
    result_kwargs['max_amino_acids_RAM'] = result_kwargs.get("max_amino_acids_RAM", 25000)

    for directory in necessary_directories:
        if not result_kwargs.get(directory):
            directory_path = file_manager.create_directory(result_kwargs.get('prefix'), result_kwargs.get('stage_name'), directory)

            get_model_directories_from_zip(
                model='albert',
                directory=directory,
                path=directory_path
            )

            result_kwargs[directory] = directory_path

    # Create reduced embeddings file if set in params
    result_kwargs['reduce'] = result_kwargs.get('reduce', False)

    reduced_embeddings_file = None
    if result_kwargs['reduce'] is True:
        reduced_embeddings_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                result_kwargs.get('stage_name'),
                                                                'reduced_embeddings_file', extension='.h5')
        result_kwargs['reduced_embeddings_file'] = reduced_embeddings_file_path
        reduced_embeddings_file = h5py.File(reduced_embeddings_file_path, "w")

    # Create embeddings file if not discarted in params
    embeddings_file = None

    result_kwargs['discard_per_amino_acid_embeddings'] = result_kwargs.get('discard_per_amino_acid_embeddings', False)

    if result_kwargs['discard_per_amino_acid_embeddings'] is True:
        if result_kwargs['reduce'] is False:
            raise InvalidParameterError(
                "Cannot have discard_per_amino_acid_embeddings=True and reduce=False. Both must be True.")
    else:
        embeddings_file_path = file_manager.create_file(result_kwargs.get('prefix'), result_kwargs.get('stage_name'),
                                                        'embeddings_file', extension='.h5')
        embeddings_file = h5py.File(embeddings_file_path, "w")
        result_kwargs['embeddings_file'] = embeddings_file_path

    # Get embedder
    embedder = AlbertEmbedder(**result_kwargs)

    # Embed iteratively based on max AA in RAM at a given time
    max_amino_acids_RAM = result_kwargs['max_amino_acids_RAM']
    protein_generator = read_fasta_file_generator(result_kwargs['remapped_sequences_file'])

    # Get sequence mapping to use as information source
    mapping_file = read_csv(result_kwargs['mapping_file'], index_col=0)

    candidates = list()
    aa_count = 0

    for _, sequence in tqdm(enumerate(protein_generator), total=len(mapping_file)):
        candidates.append(sequence)
        aa_count += len(sequence)

        if aa_count + len(sequence) > max_amino_acids_RAM:
            embeddings = embedder.embed_many([protein.seq for protein in candidates])

            for index, protein in enumerate(candidates):
                if result_kwargs.get('discard_per_amino_acid_embeddings') is False:
                    embeddings_file.create_dataset(protein.id, data=embeddings[index])

                if result_kwargs.get('reduce') is True:
                    reduced_embeddings_file.create_dataset(
                        protein.id,
                        data=embedder.reduce_per_protein(embeddings[index])
                    )

            # Reset
            aa_count = 0
            candidates = list()

    if candidates:
        embeddings = embedder.embed_many([protein.seq for protein in candidates])

        for index, protein in enumerate(candidates):
            if result_kwargs.get('discard_per_amino_acid_embeddings') is False:
                embeddings_file.create_dataset(protein.id, data=embeddings[index])

            if result_kwargs.get('reduce') is True:
                reduced_embeddings_file.create_dataset(
                    protein.id,
                    data=embedder.reduce_per_protein(embeddings[index])
                )

    # Close embeddings files
    if result_kwargs.get('discard_per_amino_acid_embeddings') is False:
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
        mapping_file: the mapping file generated by the pipeline when remapping indexes
        stage_name: The stage name

    Returns
    -------
    Dictionary with results of stage
    """
    check_required(kwargs, ['protocol', 'prefix', 'stage_name', 'remapped_sequences_file', 'mapping_file'])

    if kwargs["protocol"] not in PROTOCOLS:
        raise InvalidParameterError(
            "Invalid protocol selection: " +
            "{}. Valid protocols are: {}".format(
                kwargs["protocol"], ", ".join(PROTOCOLS.keys())
            )
        )

    return PROTOCOLS[kwargs["protocol"]](**kwargs)

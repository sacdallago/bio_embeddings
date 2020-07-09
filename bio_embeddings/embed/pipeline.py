import contextlib
import logging
from copy import deepcopy
from typing import Dict, Any

import h5py
from Bio import SeqIO
from pandas import read_csv
from tqdm import tqdm

from bio_embeddings.embed import EmbedderInterface
from bio_embeddings.embed.albert import AlbertEmbedder
from bio_embeddings.embed.seqvec.SeqVecEmbedder import SeqVecEmbedder
from bio_embeddings.utilities import (
    InvalidParameterError,
    get_model_file,
    check_required,
    get_file_manager,
    get_model_directories_from_zip,
    FileManagerInterface,
)

logger = logging.getLogger(__name__)


def _get_reduced_embeddings_file_context(file_manager: FileManagerInterface, result_kwargs: Dict):
    """
    :param file_manager: The FileManager derived class which will be used to create the file
    :param result_kwargs: A dictionary which will be updated in-place to include the path to the newly created file

    :return: a file context
    """

    # Create reduced embeddings file if set in params
    result_kwargs.setdefault('reduce', False)

    if result_kwargs['reduce'] is True:
        reduced_embeddings_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                result_kwargs.get('stage_name'),
                                                                'reduced_embeddings_file', extension='.h5')
        result_kwargs['reduced_embeddings_file'] = reduced_embeddings_file_path
        return h5py.File(reduced_embeddings_file_path, "w")

    return contextlib.nullcontext()


def _get_embeddings_file_context(
    file_manager: FileManagerInterface, result_kwargs: Dict[str, Any]
):
    """
    :param file_manager: The FileManager derived class which will be used to create the file
    :param result_kwargs: A dictionary which will be updated in-place to include the path to the newly created file

    :return: a file context
    """

    result_kwargs.setdefault("discard_per_amino_acid_embeddings", False)

    if result_kwargs["discard_per_amino_acid_embeddings"] is True:
        if result_kwargs["reduce"] is False:
            raise InvalidParameterError(
                "Cannot have discard_per_amino_acid_embeddings=True and reduce=False. Both must be True."
            )
        return contextlib.nullcontext()
    else:
        embeddings_file_path = file_manager.create_file(
            result_kwargs.get("prefix"),
            result_kwargs.get("stage_name"),
            "embeddings_file",
            extension=".h5",
        )
        result_kwargs["embeddings_file"] = embeddings_file_path
        return h5py.File(embeddings_file_path, "w")


def embed_and_write_batched(
    embedder: EmbedderInterface,
    file_manager: FileManagerInterface,
    result_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """ The shared code between the SeqVec and the Albert pipeline """
    # Lazy fasta file reader. The mapping file contains the corresponding ids in the same order
    sequences = (
        str(entry.seq)
        for entry in SeqIO.parse(result_kwargs["remapped_sequences_file"], "fasta")
    )
    mapping_file = read_csv(result_kwargs["mapping_file"], index_col=0)
    # Open embedding files or null contexts and iteratively save embeddings to file
    with _get_embeddings_file_context(
        file_manager, result_kwargs
    ) as embeddings_file, _get_reduced_embeddings_file_context(
        file_manager, result_kwargs
    ) as reduced_embeddings_file:
        embedding_generator = embedder.embed_many(
            sequences, result_kwargs["max_amino_acids"]
        )
        for sequence_id, embedding in zip(
            mapping_file.index, tqdm(embedding_generator, total=len(mapping_file))
        ):
            if result_kwargs.get("discard_per_amino_acid_embeddings") is False:
                embeddings_file.create_dataset(sequence_id, data=embedding)

            if result_kwargs.get("reduce") is True:
                reduced_embeddings_file.create_dataset(
                    sequence_id, data=embedder.reduce_per_protein(embedding),
                )
    return result_kwargs


def seqvec(**kwargs) -> Dict[str, Any]:
    necessary_files = ["weights_file", "options_file"]
    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager(**kwargs)

    # Initialize pipeline and model specific options:
    result_kwargs.setdefault("max_amino_acids", 15000)

    if result_kwargs.get("seqvec_version") == 2 or result_kwargs.get("vocabulary_file"):
        necessary_files.append("vocabulary_file")
        result_kwargs["seqvec_version"] = 2

    # Download necessary files if needed
    for file in necessary_files:
        if not result_kwargs.get(file):
            file_path = file_manager.create_file(
                result_kwargs.get("prefix"), result_kwargs.get("stage_name"), file
            )

            get_model_file(
                path=file_path,
                model="seqvecv{}".format(result_kwargs.get("seqvec_version", 1)),
                file=file,
            )

            result_kwargs[file] = file_path

    embedder = SeqVecEmbedder(**result_kwargs)
    return embed_and_write_batched(embedder, file_manager, result_kwargs)


def albert(**kwargs):
    necessary_directories = ["model_directory"]
    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager(**kwargs)

    result_kwargs.setdefault("max_amino_acids", 500)

    for directory in necessary_directories:
        if not result_kwargs.get(directory):
            directory_path = file_manager.create_directory(
                result_kwargs.get("prefix"), result_kwargs.get("stage_name"), directory
            )

            get_model_directories_from_zip(
                path=directory_path, model="albert", directory=directory
            )

            result_kwargs[directory] = directory_path

    embedder = AlbertEmbedder(**result_kwargs)
    return embed_and_write_batched(embedder, file_manager, result_kwargs)


def fasttext(**kwargs):
    pass


def glove(**kwargs):
    pass


def word2vec(**kwargs):
    pass


# list of available embedding protocols
PROTOCOLS = {
    "seqvec": seqvec,
    "fasttext": fasttext,
    "glove": glove,
    "word2vec": word2vec,
    "albert": albert,
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
            "Invalid protocol selection: {}. Valid protocols are: {}".format(
                kwargs["protocol"], ", ".join(PROTOCOLS.keys())
            )
        )

    return PROTOCOLS[kwargs["protocol"]](**kwargs)

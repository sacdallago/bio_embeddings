import logging
import shutil
from copy import deepcopy
from typing import Dict, Any, Type

import h5py
from Bio import SeqIO
from pandas import read_csv, DataFrame
from tqdm import tqdm

from bio_embeddings.embed import (
    ProtTransAlbertBFDEmbedder,
    ProtTransBertBFDEmbedder,
    EmbedderInterface,
    SeqVecEmbedder,
    ProtTransXLNetUniRef100Embedder,
    UniRepEmbedder,
)
from bio_embeddings.utilities import (
    InvalidParameterError,
    get_model_file,
    check_required,
    get_file_manager,
    get_model_directories_from_zip,
    FileManagerInterface,
)
from bio_embeddings.utilities.backports import nullcontext

logger = logging.getLogger(__name__)


def _print_expected_file_sizes(
    embedder: EmbedderInterface, mapping_file: DataFrame, result_kwargs: Dict[str, Any]
) -> None:
    """
    Logs the lower bound size of embeddings_file and reduced_embedding_file

    :param embedder: the embedder being used
    :param mapping_file: the mapping file of the sequences
    :param result_kwargs: the kwargs passed to the pipeline --> will decide what to print

    :return: Nothing.
    """
    per_amino_acid_size_in_bytes = 4 * embedder.embedding_dimension * embedder.number_of_layers
    per_protein_size_in_bytes = 4 * embedder.embedding_dimension

    total_number_of_proteins = len(mapping_file)
    total_aa = mapping_file['sequence_length'].sum()

    embeddings_file_size_in_MB = per_amino_acid_size_in_bytes * total_aa * pow(10, -6)
    reduced_embeddings_file_size_in_MB = per_protein_size_in_bytes * total_number_of_proteins * pow(10, -6)

    required_space_in_MB = 0

    if result_kwargs.get("reduce") is True:
        logger.info(f"The minimum expected size for the reduced_embedding_file is "
                    f"{reduced_embeddings_file_size_in_MB:.3f}MB.")

        required_space_in_MB += reduced_embeddings_file_size_in_MB

    if not (result_kwargs.get("reduce") is True and result_kwargs.get("discard_per_amino_acid_embeddings") is True):
        logger.info(f"The minimum expected size for the embedding_file is {embeddings_file_size_in_MB:.3f}MB.")

        required_space_in_MB += embeddings_file_size_in_MB

    _, _, available_space_in_bytes = shutil.disk_usage(result_kwargs.get('prefix'))

    available_space_in_MB = available_space_in_bytes * pow(10, -6)

    if available_space_in_MB < required_space_in_MB:
        logger.warning(f"You are attempting to generate {required_space_in_MB:.3f}MB worth of embeddings, "
                       f"but only {available_space_in_MB:.3f}MB are available at "
                       f"the prefix({result_kwargs.get('prefix')}). \n"
                       f"We suggest you stop execution NOW and double check you have enough free space available. "
                       f"Alternatively, try reducing the input FASTA file.")
    else:
        logger.info(f"You are going to generate a total of {required_space_in_MB:.3f}MB of embeddings, and have "
                    f"{available_space_in_MB:.3f}MB available at {result_kwargs.get('prefix')}.")


def _get_reduced_embeddings_file_context(
    file_manager: FileManagerInterface, result_kwargs: Dict[str, Any]
):
    """
    :param file_manager: The FileManager derived class which will be used to create the file
    :param result_kwargs: A dictionary which will be updated in-place to include the path to the newly created file

    :return: a file context
    """

    # Create reduced embeddings file if set in params
    result_kwargs.setdefault("reduce", False)

    if result_kwargs["reduce"] is True:
        reduced_embeddings_file_path = file_manager.create_file(
            result_kwargs.get("prefix"),
            result_kwargs.get("stage_name"),
            "reduced_embeddings_file",
            extension=".h5",
        )
        result_kwargs["reduced_embeddings_file"] = reduced_embeddings_file_path
        return h5py.File(reduced_embeddings_file_path, "w")

    return nullcontext()


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
        return nullcontext()
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
    """ The shared code between the SeqVec, Albert, Bert and XLNet pipelines """
    # Lazy fasta file reader. The mapping file contains the corresponding ids in the same order
    sequences = (
        str(entry.seq)
        for entry in SeqIO.parse(result_kwargs["remapped_sequences_file"], "fasta")
    )
    # We want to read the unnamed column 0 as str (esp. with simple_remapping), which requires some workarounds
    # https://stackoverflow.com/a/29793294/3549270
    mapping_file = read_csv(result_kwargs["mapping_file"], index_col=0)
    mapping_file.index = mapping_file.index.astype('str')

    # Print the minimum required file sizes
    _print_expected_file_sizes(embedder, mapping_file, result_kwargs)

    # Open embedding files or null contexts and iteratively save embeddings to file
    with _get_embeddings_file_context(
        file_manager, result_kwargs
    ) as embeddings_file, _get_reduced_embeddings_file_context(
        file_manager, result_kwargs
    ) as reduced_embeddings_file:
        embedding_generator = embedder.embed_many(
            sequences, result_kwargs.get("max_amino_acids")
        )
        for sequence_id, embedding in zip(
            mapping_file.index, tqdm(embedding_generator, total=len(mapping_file))
        ):
            if result_kwargs.get("discard_per_amino_acid_embeddings") is False:
                embeddings_file.create_dataset(sequence_id, data=embedding)

            if result_kwargs.get("reduce") is True:
                reduced_embeddings_file.create_dataset(
                    sequence_id, data=embedder.reduce_per_protein(embedding)
                )
    return result_kwargs


def seqvec(**kwargs) -> Dict[str, Any]:
    necessary_files = ["weights_file", "options_file"]
    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager(**kwargs)

    # Initialize pipeline and model specific options:
    result_kwargs.setdefault("max_amino_acids", 15000)

    # Download necessary files if needed
    for file in necessary_files:
        if not result_kwargs.get(file):
            result_kwargs[file] = get_model_file(model="seqvec", file=file)

    embedder = SeqVecEmbedder(**result_kwargs)
    return embed_and_write_batched(embedder, file_manager, result_kwargs)


def transformer(
    embedder_class: Type[EmbedderInterface], max_amino_acids_default: int, **kwargs
):
    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager(**kwargs)
    result_kwargs.setdefault("max_amino_acids", max_amino_acids_default)

    necessary_directories = ["model_directory"]
    for directory in necessary_directories:
        if not result_kwargs.get(directory):
            result_kwargs[directory] = get_model_directories_from_zip(
                model=embedder_class.name, directory=directory
            )

    embedder = embedder_class(**result_kwargs)
    return embed_and_write_batched(embedder, file_manager, result_kwargs)


def prottrans_albert(**kwargs):
    return transformer(ProtTransAlbertBFDEmbedder, 3035, **kwargs)


def prottrans_bert_bfd(**kwargs):
    return transformer(ProtTransBertBFDEmbedder, 6024, **kwargs)


def prottrans_xlnet(**kwargs):
    return transformer(ProtTransXLNetUniRef100Embedder, 4000, **kwargs)


def unirep(**kwargs) -> Dict[str, Any]:
    if kwargs.get("use_cpu") is not None:
        raise InvalidParameterError("UniRep does not support configuring `use_cpu`")
    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager(**kwargs)
    embedder = UniRepEmbedder(**result_kwargs)
    # We don't actually batch with UniRep, but embed_and_write_batched
    # works anyway since UniRepEmbedder still implements `embed_many`
    return embed_and_write_batched(embedder, file_manager, result_kwargs)


# list of available embedding protocols
PROTOCOLS = {
    "seqvec": seqvec,
    "prottrans_albert_bfd": prottrans_albert,
    "prottrans_bert_bfd": prottrans_bert_bfd,
    "prottrans_xlnet_uniref100": prottrans_xlnet,
    "unirep": unirep,
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
    check_required(
        kwargs,
        ["protocol", "prefix", "stage_name", "remapped_sequences_file", "mapping_file"],
    )

    if kwargs["protocol"] not in PROTOCOLS:
        raise InvalidParameterError(
            "Invalid protocol selection: {}. Valid protocols are: {}".format(
                kwargs["protocol"], ", ".join(PROTOCOLS.keys())
            )
        )

    return PROTOCOLS[kwargs["protocol"]](**kwargs)

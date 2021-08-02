import logging
import shutil
from copy import deepcopy
from typing import Dict, Any

import h5py
import numpy
from Bio import SeqIO
from humanize import naturalsize
from pandas import DataFrame
from tqdm import tqdm

from bio_embeddings.embed import name_to_embedder, EmbedderInterface
from bio_embeddings.utilities import (
    FileManagerInterface,
    InvalidParameterError,
    check_required,
    get_file_manager,
    get_model_directories_from_zip,
    get_model_file,
    read_mapping_file,
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

    embeddings_file_size = per_amino_acid_size_in_bytes * total_aa
    reduced_embeddings_file_size = per_protein_size_in_bytes * total_number_of_proteins

    required_space = 0

    if result_kwargs.get("reduce") is True:
        logger.info(f"The minimum expected size for the reduced_embedding_file is "
                    f"{naturalsize(reduced_embeddings_file_size)}.")

        required_space += reduced_embeddings_file_size

    # TODO: calculate size of transformed embeddings?
    if not (result_kwargs.get("reduce") is True and result_kwargs.get("discard_per_amino_acid_embeddings") is True):
        logger.info(f"The minimum expected size for the embedding_file is {naturalsize(embeddings_file_size)}.")

        required_space += embeddings_file_size

    _, _, available_space_in_bytes = shutil.disk_usage(result_kwargs.get('prefix'))

    available_space = available_space_in_bytes

    if available_space < required_space:
        logger.warning(f"You are attempting to generate {naturalsize(required_space)} worth of embeddings, "
                       f"but only {naturalsize(available_space)} are available at "
                       f"the prefix({result_kwargs.get('prefix')}). \n"
                       f"We suggest you stop execution NOW and double check you have enough free space available. "
                       f"Alternatively, try reducing the input FASTA file.")
    else:
        logger.info(f"You are going to generate a total of {naturalsize(required_space)} of embeddings, and have "
                    f"{naturalsize(available_space)} available at {result_kwargs.get('prefix')}.")


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
        if result_kwargs.get("reduce", False) is False and result_kwargs.get("embeddings_transformer_function") is None:
            raise InvalidParameterError(
                "Cannot only have discard_per_amino_acid_embeddings: True. "
                "Either also set `reduce: True` or define an `embeddings_transformer_function`, or both."
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


def _get_transformed_embeddings_file_context(
    file_manager: FileManagerInterface, result_kwargs: Dict[str, Any]
):
    """
    :param file_manager: The FileManager derived class which will be used to create the file
    :param result_kwargs: A dictionary which will be updated in-place to include the path to the newly created file

    :return: a file context
    """

    result_kwargs.setdefault("embeddings_transformer_function", None)

    if result_kwargs["embeddings_transformer_function"] is not None:
        transformed_embeddings_file_path = file_manager.create_file(
            result_kwargs.get("prefix"),
            result_kwargs.get("stage_name"),
            "transformed_embeddings_file",
            extension=".h5",
        )
        result_kwargs["transformed_embeddings_file"] = transformed_embeddings_file_path
        return h5py.File(transformed_embeddings_file_path, "w")

    return nullcontext()


def _check_transform_embeddings_function(embedder: EmbedderInterface, result_kwargs: Dict[str, Any]):
    result_kwargs.setdefault("embeddings_transformer_function", None)

    if result_kwargs["embeddings_transformer_function"] is not None:
        try:
            transform_function = eval(result_kwargs["embeddings_transformer_function"], {}, {"np": numpy})
        except TypeError:
            raise InvalidParameterError(f"`embeddings_transformer_function` must be callable! \n"
                                        f"Instead is {result_kwargs['embeddings_transformer_function']}\n"
                                        f"Most likely you want a lambda function.")

        if not callable(transform_function):
            raise InvalidParameterError(f"`embeddings_transformer_function` must be callable! \n"
                                        f"Instead is {result_kwargs['embeddings_transformer_function']}\n"
                                        f"Most likely you want a lambda function.")

        template_embedding = embedder.embed("SEQVENCE")

        # Check that it works in principle
        try:
            transformed_template_embedding = transform_function(template_embedding)
        except:
            raise InvalidParameterError(f"`embeddings_transformer_function` must be valid callable! \n"
                                        f"Instead is {result_kwargs['embeddings_transformer_function']}\n"
                                        f"This function excepts when processing an embedding.")

        # Check that return can be cast to np.array
        try:
            numpy.array(transformed_template_embedding)
        except:
            raise InvalidParameterError(f"`embeddings_transformer_function` must be valid callable "
                                        f"returning numpy array compatible object! \n"
                                        f"Instead is {result_kwargs['embeddings_transformer_function']}\n"
                                        f"This function excepts when processing an embedding.")


def embed_and_write_batched(
    embedder: EmbedderInterface,
    file_manager: FileManagerInterface,
    result_kwargs: Dict[str, Any],
    half_precision: bool = False
) -> Dict[str, Any]:
    """ The shared code between the SeqVec, Albert, Bert and XLNet pipelines """
    # Lazy fasta file reader. The mapping file contains the corresponding ids in the same order
    sequences = (
        str(entry.seq)
        for entry in SeqIO.parse(result_kwargs["remapped_sequences_file"], "fasta")
    )
    mapping_file = read_mapping_file(result_kwargs["mapping_file"])

    # Print the minimum required file sizes
    _print_expected_file_sizes(embedder, mapping_file, result_kwargs)

    # Get transformer function, if available
    transform_function = result_kwargs.get("embeddings_transformer_function", None)

    if transform_function:
        transform_function = eval(transform_function, {}, {"np": numpy})

    # Open embedding files or null contexts and iteratively save embeddings to file
    with _get_embeddings_file_context(
        file_manager, result_kwargs
    ) as embeddings_file, _get_reduced_embeddings_file_context(
        file_manager, result_kwargs
    ) as reduced_embeddings_file, _get_transformed_embeddings_file_context(
        file_manager, result_kwargs
    ) as transformed_embeddings_file:
        embedding_generator = embedder.embed_many(
            sequences, result_kwargs.get("max_amino_acids")
        )
        for sequence_id, original_id, embedding in zip(
            mapping_file.index,
            mapping_file["original_id"],
            tqdm(embedding_generator, total=len(mapping_file)),
        ):
            if half_precision:
                embedding = embedding.astype(numpy.float16)
            if result_kwargs.get("discard_per_amino_acid_embeddings") is False:
                dataset = embeddings_file.create_dataset(sequence_id, data=embedding)
                dataset.attrs["original_id"] = original_id
            if result_kwargs.get("reduce") is True:
                dataset = reduced_embeddings_file.create_dataset(
                    sequence_id, data=embedder.reduce_per_protein(embedding)
                )
                dataset.attrs["original_id"] = original_id
            if transform_function:
                dataset = transformed_embeddings_file.create_dataset(
                    sequence_id, data=numpy.array(transform_function(embedding))
                )
                dataset.attrs["original_id"] = original_id

    return result_kwargs


# Some of this might not be available when installed without the `all` extra
ALL_PROTOCOLS = [
    "bepler",
    "cpcprot",
    "esm",
    "esm1b",
    "esm1v",
    "fasttext",
    "glove",
    "one_hot_encoding",
    "plus_rnn",
    "prottrans_albert_bfd",
    "prottrans_bert_bfd",
    "prottrans_t5_bfd",
    "prottrans_t5_uniref50",
    "prottrans_xlnet_uniref100",
    "prottrans_t5_xl_u50",
    "seqvec",
    "unirep",
    "word2vec",
]

# TODO: 10000 is a random guess
# There remainder was measured for a GTX 1080 with 8GB memory
DEFAULT_MAX_AMINO_ACIDS = {
    "bepler": 10000,
    "cpcprot": 10000,
    "esm": 10000,
    "esm1b": 10000,
    "esm1v": 10000,
    "plus_rnn": 10000,
    "prottrans_albert_bfd": 3035,
    "prottrans_bert_bfd": 6024,
    "prottrans_t5_bfd": 5000,
    "prottrans_t5_uniref50": 5000,
    "prottrans_t5_xl_u50": 5000,
    "prottrans_xlnet_uniref100": 4000,
    "seqvec": 15000,
    "unirep": 10000,
    "fasttext": None,
    "glove": None,
    "one_hot_encoding": None,
    "word2vec": None,
}


def prepare_kwargs(**kwargs):
    required_kwargs = [
        "protocol",
        "prefix",
        "stage_name",
        "remapped_sequences_file",
        "mapping_file",
    ]
    check_required(kwargs, required_kwargs)

    if kwargs["protocol"] not in name_to_embedder:
        if kwargs["protocol"] in ALL_PROTOCOLS:
            raise InvalidParameterError(
                f"The extra for the protocol {kwargs['protocol']} is missing. "
                "See https://docs.bioembeddings.com/#installation on how to install all extras"
            )
        raise InvalidParameterError(
            "Invalid protocol selection: {}. Valid protocols are: {}".format(
                kwargs["protocol"], ", ".join(name_to_embedder.keys())
            )
        )

    embedder_class = name_to_embedder[kwargs["protocol"]]

    if kwargs["protocol"] == "unirep" and kwargs.get("use_cpu") is not None:
        raise InvalidParameterError("UniRep does not support configuring `use_cpu`")
    if kwargs["protocol"] == "esm1v" and not kwargs.get("ensemble_id"):
        raise InvalidParameterError(
            "You must set `ensemble_id` to select which of the five models you want to use [1-5]"
        )
    # See parameter_blueprints.yml
    global_options = {"sequences_file", "simple_remapping", "start_time"}
    embed_options = {
        "decoder",
        "device",
        "discard_per_amino_acid_embeddings",
        "ensemble_id",
        "half_precision_model",
        "half_precision",
        "max_amino_acids",
        "reduce",
        "type",
    }
    known_parameters = (
        set(required_kwargs)
        | global_options
        | embed_options
        | set(embedder_class.necessary_files)
        | set(embedder_class.necessary_directories)
    )
    if embedder_class == "seqvec":
        # We support two ways of configuration for seqvec
        known_parameters.add("model_directory")
    if not set(kwargs) < known_parameters:
        # Complain louder if the input looks fishier
        for option in set(kwargs) - known_parameters:
            logger.warning(
                f"You set an unknown option for {embedder_class.name}: {option} (value: {kwargs[option]})"
            )

    if kwargs.get("half_precision_model"):
        if kwargs["protocol"] not in ["prottrans_t5_bfd", "prottrans_t5_uniref50", "prottrans_t5_xl_u50"]:
            raise InvalidParameterError(
                "`half_precision_model` is only supported with prottrans_t5_bfd, prottrans_t5_uniref50 and prottrans_t5_xl_u50"
            )

        if kwargs.get("half_precision") is False:  # None remains allowed
            raise InvalidParameterError(
                "You can't have `half_precision_model` be true and `half_precision` be false. "
                "We suggest also setting `half_precision` to true, "
                "which will compute and save embeddings as half-precision floats"
            )

    result_kwargs = deepcopy(kwargs)
    result_kwargs.setdefault(
        "max_amino_acids", DEFAULT_MAX_AMINO_ACIDS[kwargs["protocol"]]
    )

    return embedder_class, result_kwargs


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
    embedder_class, result_kwargs = prepare_kwargs(**kwargs)

    file_manager = get_file_manager(**kwargs)
    embedder: EmbedderInterface = embedder_class(**result_kwargs)
    _check_transform_embeddings_function(embedder, result_kwargs)

    return embed_and_write_batched(embedder, file_manager, result_kwargs, kwargs.get("half_precision", False))

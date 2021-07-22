import numpy as np
import h5py
import logging
import torch

from enum import Enum
from hashlib import md5
from typing import List, Union
from pandas import DataFrame, read_csv

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from bio_embeddings.utilities.exceptions import MissingParameterError, ConversionUniqueMismatch


logger = logging.getLogger(__name__)


def get_device(device: Union[None, str, torch.device] = None) -> torch.device:
    """Returns what the user specified, or defaults to the GPU,
    with a fallback to CPU if no GPU is available."""
    if isinstance(device, torch.device):
        return device
    elif device:
        return torch.device(device)
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def check_required(params: dict, keys: List[str]):
    """
    Verify if required set of parameters is present in configuration

    Parameters
    ----------
    params : dict
        Dictionary with parameters
    keys : list-like
        Set of parameters that has to be present in params

    Raises
    ------
    MissingParameterError
    """
    missing = [k for k in keys if k not in params]

    if len(missing) > 0:
        raise MissingParameterError(
            "Missing required parameters: {} \nGiven: {}".format(
                ", ".join(missing), params
            )
        )


def _assign_hash(sequence_record: SeqRecord) -> SeqRecord:
    sequence_record.id = md5(str(sequence_record.seq).encode()).hexdigest()

    return sequence_record


def read_fasta(path: str) -> List[SeqRecord]:
    """
    Helper function to read FASTA file.

    :param path: path to a valid FASTA file
    :return: a list of SeqRecord objects.
    """
    try:
        return list(SeqIO.parse(path, "fasta"))
    except FileNotFoundError:
        raise  # Already says "No such file or directory"
    except Exception as e:
        raise ValueError(f"Could not parse '{path}'. Are you sure this is a valid fasta file?") from e


def reindex_sequences(
        sequence_records: List[SeqRecord], simple=False
) -> (SeqRecord, DataFrame):
    """
    Function will sort and re-index the sequence_records IN PLACE! (change the original list!).
    Returns a DataFrame with the mapping.

    :param sequence_records: List of sequence records
    :param simple: Boolean; if set to true use numerical index (1,2,3,4) instead of md5 hash
    :return: A dataframe with the mapping with key the new ids and a column "original_id" containing the previous id, and the sequence length.
    """
    sequence_records[:] = sorted(sequence_records, key=lambda seq: -len(seq))
    original_ids = [s.id for s in sequence_records]

    if simple:
        new_ids = list()
        for id, record in enumerate(sequence_records):
            record.id = str(id)
            new_ids.append(str(id))
    else:
        sequence_records[:] = map(_assign_hash, sequence_records)
        new_ids = [s.id for s in sequence_records]

    df = DataFrame(
        zip(original_ids, [len(seq) for seq in sequence_records]),
        columns=["original_id", "sequence_length"],
        index=new_ids,
    )

    return df


def write_fasta_file(sequence_records: List[SeqRecord], file_path: str) -> None:
    SeqIO.write(sequence_records, file_path, "fasta")


def convert_list_of_enum_to_string(list_of_enums: List[Enum]) -> str:
    return "".join([e.value for e in list_of_enums])


def reindex_h5_file(h5_file_path: str, mapping_file_path: str):
    """
    Will rename the dataset keys using the "original_id" from the mapping file.
    This operation is generally considered unsafe, as the "original_id" is unsafe
    (may contain invalid characters, duplicates, or empty strings).

    Some sanity checks are performed before starting the renaming process,
    but generally applying this function is discouraged unless you know what you are doing.

    :param h5_file_path: path to the hd5_file to re-index
    :param mapping_file_path: path to the mapping file (this must have the first column be the current keys, and a column "original_id" as the new desired id)
    :return: Nothing -- conversion happens in place!
    """

    mapping_file = read_csv(mapping_file_path, index_col=0)
    mapping_file.index = mapping_file.index.map(str)
    mapping_file['original_id'] = mapping_file['original_id'].astype(str)
    conversion_table = list(zip(mapping_file.index.values, mapping_file['original_id'].values))
    unique_froms = set([e[0] for e in conversion_table])
    unique_tos = set([e[1] for e in conversion_table if e])

    if len(unique_froms) != len(unique_tos):
        raise ConversionUniqueMismatch(f"Conversion unique count mismatch.\n"
                                       f"Your mapping file contains {len(unique_froms)} unique ids, which you are truing to convert to {len(unique_tos)} unique original_ids.\n"
                                       f"These numbers *must* match. You likely have: duplicate original_id's, or empty strings in original_id.")

    with h5py.File(h5_file_path, "r+") as h5_file:
        keys_set = set(h5_file.keys())
        unchanged_set = keys_set - unique_froms

        if len(unchanged_set) > 0:
            logger.warning(f"There are some keys in your h5 file which won't be re-indexed!\n"
                           f"These are: {unchanged_set}.")

        changeable_set = unique_froms.union(keys_set)

        if len(changeable_set) == 0:
            logger.info("Nothing was re-indexed.")

        else:
            logger.info(f"Reindexing the following keys: {changeable_set}")

            for (from_index, to_index) in filter(lambda item: item[0] in keys_set, conversion_table):
                h5_file.move(from_index, to_index)


def remove_identifiers_from_annotations_file(faulty_identifiers: list, annotation_file_path: str) -> DataFrame:
    """
    Removes id
    :param faulty_identifiers: a list of identifiers
    :param annotation_file_path: a str detailing the path
    :return: a new DataFrame with the annotations removed
    """

    annotation_file = read_csv(annotation_file_path)
    return annotation_file[annotation_file['identifier'].isin(
        set(annotation_file['identifier'].values) - set(faulty_identifiers)
    )]


class QueryEmbeddingsFile:
    """
    A helper class that allows you to retrieve embeddings from an embeddings file based on either the `original_id`
    (extracted from the FASTA header during the embed stage), or via the `new_id` (assigned during the embed stage,
    either an MD5 hash of the input sequence, or an integer (if `remapping_simple: True`).

    Available for embeddings created with the pipeline starting with v0.1.5

    .. code-block:: python

        import h5py
        from bio_embeddings.utilities import QueryEmbeddingsFile

        with h5py.File("path/to/file.h5", "r") as file:
            embedding_querier = QueryEmbeddingsFile(file)
            print(embedding_querier.query_original_id("Some_Database_ID_1234").mean())
    """

    def __init__(self, embeddings_file: h5py.File):
        """
        :param embeddings_file: an h5py File, aka `h5py.File("/path/to/file.h5")`.
        """
        self._lookup_table = dict(
            (embeddings_file[new_id].attrs["original_id"], new_id) for new_id in embeddings_file.keys()
        )

        self._embeddings_file = embeddings_file

    def query_original_id(self, original_id: str) -> np.array:
        """
        Query embeddings file using the original id, aka. the string extracted from the FASTA header of the sequence.

        :param original_id: a string representing the id extracted from the FASTA header
        :return: the embedding as a numpy array
        """
        return np.array(self._embeddings_file[self._lookup_table[original_id]])

    def query_new_id(self, new_id: str) -> np.array:
        """
        Query embeddings file using the new id, aka. either the MD5 hash of the sequence or a number.

        :param new_id: a string representing the new id.
        :return: the embedding as a numpy array
        """
        return np.array(self._embeddings_file[new_id])


def read_mapping_file(mapping_file: str) -> DataFrame:
    """Reads mapping_file.csv and ensures consistent types"""
    # We want to read the unnamed column 0 as str (esp. with simple_remapping), which requires some workarounds
    # https://stackoverflow.com/a/29793294/3549270
    mapping_file = read_csv(mapping_file, index_col=0)
    mapping_file.index = mapping_file.index.astype("str")
    return mapping_file

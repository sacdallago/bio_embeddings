from enum import Enum
from hashlib import md5
from typing import List, Union

import torch
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from pandas import DataFrame

from bio_embeddings.utilities.exceptions import MissingParameterError


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
    return list(SeqIO.parse(path, "fasta"))


def reindex_sequences(
    sequence_records: List[SeqRecord], simple=False
) -> (SeqRecord, DataFrame):
    """
    Function will sort and re-index the sequence_records IN PLACE! (change the original list!).
    Returns a DataFrame with the mapping.

    :param sequence_records: List of sequence records
    :param simple: Bolean; if set to true use numerical index (1,2,3,4) instead of md5 hash
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

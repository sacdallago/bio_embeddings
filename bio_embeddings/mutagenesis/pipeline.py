import math
from copy import deepcopy
from typing import List, Dict, Type

import torch
from Bio import SeqIO
from pandas import DataFrame
from tqdm import tqdm

from bio_embeddings.mutagenesis.constants import PROBABILITIES_COLUMNS

try:
    from bio_embeddings.mutagenesis.protbert_bfd import ProtTransBertBFDMutagenesis

except ImportError as e:

    class ProtTransBertBFDMutagenesis:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"The 'transformers' required for protbert_bfd_mutagenesis is missing. "
                "See https://docs.bioembeddings.com/#installation on how to install all extras"
            ) from e


from bio_embeddings.utilities import check_required, get_device, get_file_manager, read_mapping_file

# list of available mutagenesis protocols
_PROTOCOLS = {
    "protbert_bfd_mutagenesis": ProtTransBertBFDMutagenesis,
}


def probabilities_as_dataframe(
    mapping_file: DataFrame,
    probabilities_all: Dict[str, List[Dict[str, float]]],
    sequences: List[str],
) -> DataFrame:
    """Let's build a csv with all the data"""
    records = []
    for sequence, (sequence_id, probabilities) in zip(
        sequences, probabilities_all.items()
    ):
        for wild_type_amino_acid, position_probabilities in zip(
            sequence, probabilities
        ):
            records.append(
                {
                    "id": sequence_id,
                    "original_id": mapping_file.loc[sequence_id]["original_id"],
                    "wild_type_amino_acid": wild_type_amino_acid,
                    **position_probabilities,
                }
            )
    return DataFrame(records, columns=PROBABILITIES_COLUMNS)


def run(**kwargs):
    """BETA: in-silico mutagenesis using BertForMaskedLM

    optional (see extract stage for details):
     * model_directory
     * device
     * half_precision
     * half_precision_model
     * temperature: temperature for softmax
    """
    required_kwargs = [
        "protocol",
        "prefix",
        "stage_name",
        "remapped_sequences_file",
        "mapping_file",
    ]
    check_required(kwargs, required_kwargs)
    result_kwargs = deepcopy(kwargs)
    if result_kwargs["protocol"] not in _PROTOCOLS:
        raise RuntimeError(
            f"Passed protocol {result_kwargs['protocol']}, but allowed are: {', '.join(_PROTOCOLS)}"
        )
    temperature = result_kwargs.setdefault("temperature", 1)
    device = get_device(result_kwargs.get("device"))
    model_class: Type[ProtTransBertBFDMutagenesis] = _PROTOCOLS[
        result_kwargs["protocol"]
    ]
    model = model_class(
        device,
        result_kwargs.get("model_directory"),
        result_kwargs.get("half_precision_model"),
    )

    file_manager = get_file_manager()
    file_manager.create_stage(result_kwargs["prefix"], result_kwargs["stage_name"])

    # The mapping file contains the corresponding ids in the same order
    sequences = [
        str(entry.seq)
        for entry in SeqIO.parse(result_kwargs["remapped_sequences_file"], "fasta")
    ]
    mapping_file = read_mapping_file(result_kwargs["mapping_file"])

    probabilities_all = dict()
    with tqdm(total=int(mapping_file["sequence_length"].sum())) as progress_bar:
        for sequence_id, original_id, sequence in zip(
            mapping_file.index, mapping_file["original_id"], sequences
        ):
            with torch.no_grad():
                probabilities = model.get_sequence_probabilities(
                    sequence, temperature, progress_bar=progress_bar
                )

            for p in probabilities:
                assert math.isclose(
                    1, (sum(p.values()) - p["position"]), rel_tol=1e-6
                ), "softmax values should add up to 1"

            probabilities_all[sequence_id] = probabilities
    residue_probabilities = probabilities_as_dataframe(
        mapping_file, probabilities_all, sequences
    )

    probabilities_file = file_manager.create_file(
        result_kwargs.get("prefix"),
        result_kwargs.get("stage_name"),
        "residue_probabilities_file",
        extension=".csv",
    )
    residue_probabilities.to_csv(probabilities_file, index=False)
    result_kwargs["residue_probabilities_file"] = probabilities_file
    return result_kwargs

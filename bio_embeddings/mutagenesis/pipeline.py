import math
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Type

import torch
from Bio import SeqIO
from pandas import read_csv, DataFrame
from tqdm import tqdm

from bio_embeddings.mutagenesis.constants import PROBABILITIES_COLUMNS
from bio_embeddings.mutagenesis.protbert_bfd import ProtTransBertBFDMutagenesis
from bio_embeddings.utilities import check_required, get_device, get_file_manager

# list of available mutagenesis protocols
PROTOCOLS = {
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

    optional:
     * model_directory
     * device
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
    if result_kwargs["protocol"] not in PROTOCOLS:
        raise RuntimeError(
            f"The only supported protocol is 'protbert_bfd_mutagenesis', not '{result_kwargs['protocol']}'"
        )
    temperature = result_kwargs.setdefault("temperature", 1)
    device = get_device(result_kwargs.get("device"))
    model_class: Type[ProtTransBertBFDMutagenesis] = PROTOCOLS[
        result_kwargs["protocol"]
    ]
    model = model_class(device, result_kwargs.get("model_directory"))

    file_manager = get_file_manager()
    stage = file_manager.create_stage(
        result_kwargs["prefix"], result_kwargs["stage_name"]
    )

    # The mapping file contains the corresponding ids in the same order
    sequences = [
        str(entry.seq)
        for entry in SeqIO.parse(result_kwargs["remapped_sequences_file"], "fasta")
    ]
    # We want to read the unnamed column 0 as str (esp. with simple_remapping), which requires some workarounds
    # https://stackoverflow.com/a/29793294/3549270
    mapping_file = read_csv(result_kwargs["mapping_file"], index_col=0)
    mapping_file.index = mapping_file.index.astype("str")

    probabilities_all = dict()
    with tqdm(total=int(mapping_file["sequence_length"].sum())) as progress_bar:
        for sequence_id, original_id, sequence in zip(
            mapping_file.index, mapping_file["original_id"], sequences
        ):
            with torch.no_grad():
                probabilities = model.get_sequence_probabilities(
                    sequence, temperature, None, None, progress_bar
                )

            for p in probabilities:
                assert math.isclose(
                    1, (sum(p.values()) - p["position"]), rel_tol=1e-6
                ), "softmax values should add up to 1"

            probabilities_all[sequence_id] = probabilities
    df = probabilities_as_dataframe(mapping_file, probabilities_all, sequences)

    probabilities_file = str(Path(stage).joinpath("probabilities.csv"))
    df.to_csv(probabilities_file, index=False)
    result_kwargs["probabilities_file"] = probabilities_file
    return result_kwargs

import math
from copy import deepcopy
from pathlib import Path

import torch
from Bio import SeqIO
from pandas import read_csv, DataFrame
from tqdm import tqdm

from bio_embeddings.mutagenesis.constants import PROBABILITIES_COLUMNS
from bio_embeddings.mutagenesis.protbert_bfd import (
    get_model,
    get_sequence_probabilities,
)
from bio_embeddings.utilities import check_required, get_device, get_file_manager


def run(**kwargs):
    """WIP Do Not Use

    optional:
     * model_directory
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
    if result_kwargs["protocol"] != "protbert_bfd_mutagenesis":
        raise RuntimeError(
            f"The only supported protocol is 'protbert_bfd_mutagenesis', not '{result_kwargs['protocol']}'"
        )
    result_kwargs.setdefault("temperature", 1)
    file_manager = get_file_manager()
    stage = file_manager.create_stage(
        result_kwargs["prefix"], result_kwargs["stage_name"]
    )

    device = get_device(result_kwargs.get("device"))
    tokenizer, model = get_model(device, result_kwargs.get("model_directory"))

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
    with tqdm(total=int(mapping_file["sequence_length"].sum())) as pbar:
        for sequence_id, original_id, sequence in zip(
            mapping_file.index,
            mapping_file["original_id"],
            sequences,
        ):
            with torch.no_grad():
                probabilities = get_sequence_probabilities(
                    sequence, tokenizer, model, device, result_kwargs["temperature"], None, None, pbar
                )

            for p in probabilities:
                assert math.isclose(
                    1, (sum(p.values()) - p["position"]), rel_tol=1e-6
                ), "softmax values should add up to 1"

            probabilities_all[sequence_id] = probabilities

    # Let's build a csv with all the data
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
    df = DataFrame(records, columns=PROBABILITIES_COLUMNS)
    probabilities_file = str(Path(stage).joinpath("probabilities.csv"))
    df.to_csv(probabilities_file, index=False)
    result_kwargs["probabilities_file"] = probabilities_file
    return result_kwargs

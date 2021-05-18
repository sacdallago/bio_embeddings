from copy import deepcopy
from pathlib import Path

import plotly
import torch
from Bio import SeqIO
from pandas import read_csv
from tqdm import tqdm

from bio_embeddings.mutagenesis.protbert_bfd import (
    get_model,
    get_sequence_probabilities,
    plot,
)
from bio_embeddings.utilities import check_required, get_device, get_file_manager


def run(**kwargs):
    """WIP Do Not Use

    optional:
     * model_directory
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
    file_manager = get_file_manager()
    stage = file_manager.create_stage(
        result_kwargs["prefix"], result_kwargs["stage_name"]
    )

    device = get_device(result_kwargs.get("device"))
    tokenizer, model = get_model(device, result_kwargs.get("model_directory"))

    # Lazy fasta file reader. The mapping file contains the corresponding ids in the same order
    sequences = (
        str(entry.seq)
        for entry in SeqIO.parse(result_kwargs["remapped_sequences_file"], "fasta")
    )
    # We want to read the unnamed column 0 as str (esp. with simple_remapping), which requires some workarounds
    # https://stackoverflow.com/a/29793294/3549270
    mapping_file = read_csv(result_kwargs["mapping_file"], index_col=0)
    mapping_file.index = mapping_file.index.astype("str")

    with tqdm(total=int(mapping_file["sequence_length"].sum())) as pbar:
        for sequence_id, original_id, sequence in zip(
            mapping_file.index,
            mapping_file["original_id"],
            sequences,
        ):
            with torch.no_grad():
                probabilities = get_sequence_probabilities(
                    sequence, tokenizer, model, device, None, None, pbar
                )

            fig = plot(sequence, probabilities, original_id, None, None)
            plotly.offline.plot(
                fig, filename=str(Path(stage).joinpath(f"{sequence_id}.html"))
            )
            plotly.offline.plot(fig, filename=f"{sequence_id}.html")

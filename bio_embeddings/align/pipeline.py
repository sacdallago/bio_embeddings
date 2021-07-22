import itertools
import re
from copy import deepcopy
from itertools import zip_longest
from typing import Dict, Any, List, Tuple

import torch
from Bio import SeqIO
from pandas import read_csv
from slugify import slugify

from bio_embeddings.align import deepblast_align
from bio_embeddings.utilities import (
    get_model_file,
    get_file_manager,
    MissingParameterError,
    read_fasta,
)
from bio_embeddings.utilities.exceptions import InvalidParameterError
from bio_embeddings.utilities.helpers import check_required, read_mapping_file


def pairwise_alignments_to_msa(
    queries_aligned: List[str], targets_aligned: List[str]
) -> Tuple[str, List[str]]:
    """Combines multiple alignments with a query into an MSA by padding with gaps"""

    padded_query = ""
    padded_targets = [""] * len(targets_aligned)

    # We break the sequence into each query residue plus following gaps
    query_bases = [
        list(re.finditer("[^-]-*", query_aligned)) for query_aligned in queries_aligned
    ]

    # Iterate over each residue plus gaps for all aligned query sequence
    for matches in zip_longest(*query_bases):
        # Ensures we're always matching the correct character
        [query_amino_acid] = list(set(match[0][0] for match in matches))
        # Largest number of residues in any alignment aligned to the current query residue or following gaps
        total_len = max(len(match[0]) for match in matches)
        padded_query += query_amino_acid + ("-" * (total_len - len(query_amino_acid)))
        for index, (match, target) in enumerate(zip(matches, targets_aligned)):
            # Get the residues aligned to the query residue or following gaps
            target_aligned = target[match.span()[0] : match.span()[1]]
            padded_targets[index] += target_aligned + (
                "-" * (total_len - len(target_aligned))
            )

    # Ensure the MSA is valid length wise
    [target_len] = list(set(len(padded_target) for padded_target in padded_targets))
    assert target_len == len(padded_query), "Inconsistent alignment lengths"

    return padded_query, padded_targets


def deepblast(**kwargs) -> Dict[str, Any]:
    """Sequence-Sequence alignments with DeepBLAST

    DeepBLAST learned structural alignments from sequence

    https://github.com/flatironinstitute/deepblast

    https://www.biorxiv.org/content/10.1101/2020.11.03.365932v1
    """
    # TODO: Fix that logic before merging
    if "transferred_annotations_file" not in kwargs and "pairings_file" not in kwargs:
        raise MissingParameterError(
            "You need to specify either 'transferred_annotations_file' or 'pairings_file' for DeepBLAST"
        )
    if "transferred_annotations_file" in kwargs and "pairings_file" in kwargs:
        raise InvalidParameterError(
            "You can't specify both 'transferred_annotations_file' and 'pairings_file' for DeepBLAST"
        )
    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager(**kwargs)

    # This stays below 8GB, so it should be a good default
    batch_size = result_kwargs.setdefault("batch_size", 50)

    if "device" in result_kwargs:
        device = torch.device(result_kwargs["device"])
        if device.type != "cuda":
            raise RuntimeError(
                f"You can only run DeepBLAST on a CUDA-compatible GPU, not on {device.type}"
            )
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DeepBLAST requires a CUDA-compatible GPU, but none was found"
            )
        device = torch.device("cuda")

    mapping_file = read_mapping_file(result_kwargs["mapping_file"])
    mapping = {
        str(remapped): original
        for remapped, original in mapping_file[["original_id"]].itertuples()
    }

    query_by_id = {
        mapping[entry.name]: str(entry.seq)
        for entry in SeqIO.parse(result_kwargs["remapped_sequences_file"], "fasta")
    }

    # You can either provide a set of pairing or use the output of k-nn with a fasta file for the reference embeddings
    if "pairings_file" in result_kwargs:
        pairings_file = read_csv(result_kwargs["pairings_file"])
        pairings = list(pairings_file[["query", "target"]].itertuples(index=False))
        target_by_id = query_by_id
    else:
        transferred_annotations_file = read_csv(
            result_kwargs["transferred_annotations_file"]
        )
        pairings = []
        for _, row in transferred_annotations_file.iterrows():
            query = row["original_id"]
            for target in row.filter(regex="k_nn_.*_identifier"):
                pairings.append((query, target))

        target_by_id = {}
        for entry in read_fasta(result_kwargs["reference_fasta_file"]):
            target_by_id[entry.name] = str(entry.seq[:])

    # Create one output file per query
    result_kwargs["alignment_files"] = dict()
    for query in set(i for i, _ in pairings):
        filename = file_manager.create_file(
            result_kwargs.get("prefix"),
            result_kwargs.get("stage_name"),
            f"{slugify(query, lowercase=False)}_alignments",
            extension=".a2m",
        )
        result_kwargs["alignment_files"][query] = filename

    unknown_queries = set(list(zip(*pairings))[0]) - set(query_by_id.keys())
    if unknown_queries:
        raise ValueError(f"Unknown query sequences: {unknown_queries}")

    unknown_targets = set(list(zip(*pairings))[1]) - set(target_by_id.keys())
    if unknown_targets:
        raise ValueError(f"Unknown target sequences: {unknown_targets}")

    # Load the pretrained model
    if "model_file" not in result_kwargs:
        model_file = get_model_file("deepblast", "model_file")
    else:
        model_file = result_kwargs["model_file"]

    alignments = deepblast_align(
        pairings, query_by_id, target_by_id, model_file, device, batch_size
    )

    for query, alignments in itertools.groupby(alignments, key=lambda i: i[0]):
        _, targets, queries_aligned, targets_aligned = list(zip(*alignments))
        padded_query, padded_targets = pairwise_alignments_to_msa(
            queries_aligned, targets_aligned
        )
        with open(result_kwargs["alignment_files"][query], "w") as fp:
            fp.write(f">{query}\n")
            fp.write(f"{padded_query}\n")
            for target, padded_target in zip(targets, padded_targets):
                fp.write(f">{target}\n")
                fp.write(f"{padded_target}\n")

    return result_kwargs


# list of available alignment protocols
PROTOCOLS = {
    "deepblast": deepblast,
}


def run(**kwargs):
    """
    Align query sequences with target sequences

    Parameters
    ----------
    kwargs arguments (* denotes optional):
        prefix: Output prefix for all generated files
        stage_name: The stage name
        protocol: Which method to use for alignment
        mapping_file: The mapping file generated by the pipeline when remapping indexes
        remapped_sequences_file: The fasta file with entries corresponding to the mapping file

    Returns
    -------
    Dictionary with results of stage
    """
    check_required(
        kwargs,
        [
            "protocol",
            "prefix",
            "stage_name",
            "remapped_sequences_file",
            "mapping_file",
        ],
    )

    if kwargs["protocol"] not in PROTOCOLS:
        raise InvalidParameterError(
            "Invalid protocol selection: "
            + "{}. Valid protocols are: {}".format(
                kwargs["protocol"], ", ".join(PROTOCOLS.keys())
            )
        )

    return PROTOCOLS[kwargs["protocol"]](**kwargs)

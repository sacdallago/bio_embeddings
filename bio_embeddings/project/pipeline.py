from copy import deepcopy
from typing import Dict, Any

import h5py
import numpy as np
from pandas import DataFrame

from bio_embeddings.project.pb_tucker import PBTucker
from bio_embeddings.project.tsne import tsne_reduce
from bio_embeddings.project.umap import umap_reduce
from bio_embeddings.utilities import (
    InvalidParameterError,
    check_required,
    get_file_manager,
    FileManagerInterface,
    get_model_file,
    get_device,
    read_mapping_file,
)


def write_embeddings(
    mapping: DataFrame,
    projected_embeddings: np.ndarray,
    result_kwargs: Dict[str, Any],
    file_manager: FileManagerInterface,
):
    """Writes t-sne/umap to both the legacy csv and the new h5 format"""
    # Write old csv file
    projected_reduced_embeddings_file_path = file_manager.create_file(
        result_kwargs.get("prefix"),
        result_kwargs.get("stage_name"),
        "projected_embeddings_file",
        extension=".csv",
    )

    for i in range(result_kwargs["n_components"]):
        mapping[f"component_{i}"] = projected_embeddings[:, i]

    mapping.to_csv(projected_reduced_embeddings_file_path)

    # Write new h5 file
    projected_reduced_embeddings_file_path = file_manager.create_file(
        result_kwargs.get("prefix"),
        result_kwargs.get("stage_name"),
        "projected_reduced_embeddings_file",
        extension=".h5",
    )

    with h5py.File(projected_reduced_embeddings_file_path, "w") as output_embeddings:
        for (sequence_id, original_id), projected_embedding in zip(
            mapping[["original_id"]].itertuples(), projected_embeddings
        ):
            dataset = output_embeddings.create_dataset(
                sequence_id, data=projected_embedding
            )
            dataset.attrs["original_id"] = original_id

    result_kwargs[
        "projected_reduced_embeddings_file"
    ] = projected_reduced_embeddings_file_path


def tsne(file_manager: FileManagerInterface, result_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    # Get sequence mapping to use as information source
    mapping = read_mapping_file(result_kwargs["mapping_file"])

    reduced_embeddings_file_path = result_kwargs['reduced_embeddings_file']

    reduced_embeddings = []

    with h5py.File(reduced_embeddings_file_path, 'r') as f:
        for remapped_id in mapping.index:
            reduced_embeddings.append(np.array(f[str(remapped_id)]))

    # Get parameters or set defaults
    result_kwargs.setdefault('perplexity', 6)
    result_kwargs.setdefault('n_jobs', -1)
    result_kwargs.setdefault('n_iter', 15000)
    result_kwargs.setdefault('metric', 'cosine')
    result_kwargs.setdefault('n_components', 3)
    result_kwargs.setdefault('random_state', 420)
    result_kwargs.setdefault('verbose', 1)

    projected_embeddings = tsne_reduce(reduced_embeddings, **result_kwargs)

    write_embeddings(mapping, projected_embeddings, result_kwargs, file_manager)

    return result_kwargs


def umap(file_manager: FileManagerInterface, result_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    # Get sequence mapping to use as information source
    mapping = read_mapping_file(result_kwargs["mapping_file"])

    reduced_embeddings_file_path = result_kwargs['reduced_embeddings_file']

    reduced_embeddings = []

    with h5py.File(reduced_embeddings_file_path, 'r') as f:
        for remapped_id in mapping.index:
            reduced_embeddings.append(np.array(f[str(remapped_id)]))

    # Get parameters or set defaults
    result_kwargs.setdefault('min_dist', .6)
    result_kwargs.setdefault('n_neighbors', 15)
    result_kwargs.setdefault('spread', 1)
    result_kwargs.setdefault('metric', 'cosine')
    result_kwargs.setdefault('n_components', 3)
    result_kwargs.setdefault('random_state', 420)
    result_kwargs.setdefault('verbose', 1)

    projected_embeddings = umap_reduce(reduced_embeddings, **result_kwargs)

    write_embeddings(mapping, projected_embeddings, result_kwargs, file_manager)

    return result_kwargs


def pb_tucker(
    file_manager: FileManagerInterface, result_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    device = get_device(result_kwargs.get("device"))

    if "model_file" not in result_kwargs:
        model_file = get_model_file("pb_tucker", "model_file")
    else:
        model_file = result_kwargs["model_file"]
    pb_tucker = PBTucker(model_file, device)

    reduced_embeddings_file_path = result_kwargs["reduced_embeddings_file"]
    projected_reduced_embeddings_file_path = file_manager.create_file(
        result_kwargs.get("prefix"),
        result_kwargs.get("stage_name"),
        "projected_reduced_embeddings_file",
        extension=".csv",
    )
    result_kwargs[
        "projected_reduced_embeddings_file"
    ] = projected_reduced_embeddings_file_path

    with h5py.File(reduced_embeddings_file_path, "r") as input_embeddings, h5py.File(
        projected_reduced_embeddings_file_path, "w"
    ) as output_embeddings:
        for h5_id, reduced_embedding in input_embeddings.items():
            output_embeddings[h5_id] = pb_tucker.project_reduced_embedding(reduced_embedding)

    return result_kwargs


# list of available projection protocols
PROTOCOLS = {
    "tsne": tsne,
    "umap": umap,
    "pb_tucker": pb_tucker,
}


def run(**kwargs):
    """
    Run project protocol

    Parameters
    ----------
    kwargs arguments (* denotes optional):
        projected_reduced_embeddings_file or projected_embeddings_file or reduced_embeddings_file: Where per-protein embeddings live
        prefix: Output prefix for all generated files
        stage_name: The stage name
        protocol: Which projection technique to use
        mapping_file: the mapping file generated by the pipeline when remapping indexes

    Returns
    -------
    Dictionary with results of stage
    """
    check_required(kwargs, ["protocol", "prefix", "stage_name", "mapping_file"])

    if kwargs["protocol"] not in PROTOCOLS:
        raise InvalidParameterError(
            "Invalid protocol selection: "
            + "{}. Valid protocols are: {}".format(
                kwargs["protocol"], ", ".join(PROTOCOLS.keys())
            )
        )

    result_kwargs = deepcopy(kwargs)

    # We want to allow chaining protocols, e.g. first tucker than umap,
    # so we need to allow projected embeddings as input
    embeddings_input_file = (
        kwargs.get("projected_reduced_embeddings_file")
        or kwargs.get("projected_embeddings_file")
        or kwargs.get("reduced_embeddings_file")
    )
    if not embeddings_input_file:
        raise InvalidParameterError(
            f"You need to provide either projected_reduced_embeddings_file or projected_embeddings_file or "
            f"reduced_embeddings_file for {kwargs['protocol']}"
        )
    result_kwargs["reduced_embeddings_file"] = embeddings_input_file

    file_manager = get_file_manager(**kwargs)

    result_kwargs = PROTOCOLS[kwargs["protocol"]](file_manager, result_kwargs)

    return result_kwargs

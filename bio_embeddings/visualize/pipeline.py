import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import h5py
from pandas import read_csv, DataFrame

from bio_embeddings.utilities import InvalidParameterError, check_required, get_file_manager, TooFewComponentsException
from bio_embeddings.visualize import render_3D_scatter_plotly, render_scatter_plotly, \
    save_plotly_figure_to_html
from bio_embeddings.visualize.mutagenesis import plot_mutagenesis

logger = logging.getLogger(__name__)


def plotly(result_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    file_manager = get_file_manager(**result_kwargs)

    # 2 or 3D plot? Usually, this is directly fetched from "project" stage via "depends_on"
    result_kwargs['n_components'] = result_kwargs.get('n_components', 3)
    if result_kwargs['n_components'] < 2:
        raise TooFewComponentsException(f"n_components is set to {result_kwargs['n_components']}. It should be >1.\n"
                                        f"If set to 2, will render 2D scatter plot.\n"
                                        f"If set to >3, will render 3D scatter plot.")
    # Get projected_embeddings_file containing x,y,z coordinates and identifiers
    suffix = Path(result_kwargs["projected_reduced_embeddings_file"]).suffix
    if suffix == ".csv":
        # Support the legacy csv format
        merged_annotation_file = read_csv(
            result_kwargs["projected_reduced_embeddings_file"], index_col=0
        )
    elif suffix == ".h5":
        # convert h5 to dataframe with ids and one column per dimension
        rows = []
        with h5py.File(result_kwargs["projected_reduced_embeddings_file"], "r") as file:
            for sequence_id, embedding in file.items():
                if result_kwargs['n_components'] > 2 and embedding.shape != (3,):
                    raise RuntimeError(
                        f"Expected embeddings in projected_reduced_embeddings_file "
                        f"to be of shape (3,), not {embedding.shape}"
                    )
                row = [
                    sequence_id,
                    embedding.attrs["original_id"],
                    embedding[0],
                    embedding[1],
                ]
                if result_kwargs['n_components'] > 2:
                    row.append(embedding[2])
                rows.append(row)
        columns = [
            "sequence_id",
            "original_id",
            "component_0",
            "component_1",
        ]
        if result_kwargs['n_components'] > 2:
            columns.append("component_2")
        merged_annotation_file = DataFrame.from_records(
            rows, index="sequence_id", columns=columns
        )
    else:
        raise InvalidParameterError(
            f"Expected .csv or .h5 as suffix for projected_reduced_embeddings_file, got {suffix}"
        )

    if result_kwargs.get('annotation_file'):

        annotation_file = read_csv(result_kwargs['annotation_file']).set_index('identifier')

        # Save a copy of the annotation file with index set to identifier
        input_annotation_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                              result_kwargs.get('stage_name'),
                                                              'input_annotation_file',
                                                              extension='.csv')

        annotation_file.to_csv(input_annotation_file_path)

        # Merge annotation file and projected embedding file based on index or original id?
        result_kwargs['merge_via_index'] = result_kwargs.get('merge_via_index', False)

        # Display proteins with unknown annotation?
        result_kwargs['display_unknown'] = result_kwargs.get('display_unknown', True)

        if result_kwargs['merge_via_index']:
            if result_kwargs['display_unknown']:
                merged_annotation_file = annotation_file.join(merged_annotation_file, how="outer")
                merged_annotation_file['label'].fillna('UNKNOWN', inplace=True)
            else:
                merged_annotation_file = annotation_file.join(merged_annotation_file)
        else:
            if result_kwargs['display_unknown']:
                merged_annotation_file = annotation_file.join(merged_annotation_file.set_index('original_id'),
                                                              how="outer")
                merged_annotation_file['label'].fillna('UNKNOWN', inplace=True)
            else:
                merged_annotation_file = annotation_file.join(merged_annotation_file.set_index('original_id'))
    else:
        merged_annotation_file['label'] = 'UNKNOWN'

    merged_annotation_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                           result_kwargs.get('stage_name'),
                                                           'merged_annotation_file',
                                                           extension='.csv')

    merged_annotation_file.to_csv(merged_annotation_file_path)
    result_kwargs['merged_annotation_file'] = merged_annotation_file_path

    if result_kwargs['n_components'] == 2:
        figure = render_scatter_plotly(merged_annotation_file)
    else:
        figure = render_3D_scatter_plotly(merged_annotation_file)

    plot_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                              result_kwargs.get('stage_name'),
                                              'plot_file',
                                              extension='.html')

    save_plotly_figure_to_html(figure, plot_file_path)
    result_kwargs['plot_file'] = plot_file_path

    return result_kwargs


# list of available projection protocols
PROTOCOLS = {
    "plotly": plotly,
    "plot_mutagenesis": plot_mutagenesis,
}


def run(**kwargs):
    """
    Run visualize protocol

    Parameters
    ----------
    kwargs arguments (* denotes optional):
        projected_reduced_embeddings_file: A csv with columns: (index), original_id, x, y, z
        prefix: Output prefix for all generated files
        stage_name: The stage name
        protocol: Which plot to generate

    For plotly:
        projected_reduced_embeddings_file: The projected (dimensionality reduced) embeddings, normally coming from the project stage
        annotation_file: csv file with annotations
        display_unknown: Hide proteins for which there is no annotation in the annotation file (only relevant if annotation file is provided)
        merge_via_index: Set to True if in annotation_file identifiers correspond to sequence MD5 hashes
        n_components: 2D vs 3D plot

    For plot_mutagenesis:
        residue_probabilities_file: The csv with the probabilities, normally coming from the mutagenesis stage

    Returns
    -------
    Dictionary with results of stage
    """
    check_required(kwargs, ['protocol', 'prefix', 'stage_name'])

    if kwargs["protocol"] not in PROTOCOLS:
        raise InvalidParameterError(
            "Invalid protocol selection: " +
            "{}. Valid protocols are: {}".format(
                kwargs["protocol"], ", ".join(PROTOCOLS.keys())
            )
        )

    result_kwargs = deepcopy(kwargs)

    if kwargs["protocol"] == "plotly":
        # Support legacy projected_embeddings_file
        projected_reduced_embeddings_file = (
            kwargs.get("projected_reduced_embeddings_file")
            or kwargs.get("projected_embeddings_file")
        )
        if not projected_reduced_embeddings_file:
            raise InvalidParameterError(
                f"You need to provide either projected_reduced_embeddings_file or projected_embeddings_file or "
                f"reduced_embeddings_file for {kwargs['protocol']}"
            )
        result_kwargs["projected_reduced_embeddings_file"] = projected_reduced_embeddings_file

    return PROTOCOLS[kwargs["protocol"]](result_kwargs)

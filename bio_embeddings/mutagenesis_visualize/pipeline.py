from copy import deepcopy
from pathlib import Path

import pandas
import plotly
from pandas import DataFrame
from plotly import express as px, graph_objects as go
from plotly.graph_objs import Figure
from tqdm import tqdm

from bio_embeddings.mutagenesis import PROBABILITIES_COLUMNS, AMINO_ACIDS
from bio_embeddings.utilities import check_required, get_file_manager


def plot(probabilities: DataFrame) -> Figure:
    """Given the DataFrame from the previous stage and returns a heatmap"""
    x_labels = list(
        probabilities["position"].astype(str)
        + " "
        + probabilities["wild_type_amino_acid"]
    )

    # Only the probabilities for the amino acids
    values = probabilities[list(AMINO_ACIDS)].values.T

    fig = px.imshow(
        values,
        labels=dict(x="WT sequence", y="AA", color="Probability"),
        color_continuous_scale="blues",
        x=x_labels,
        y=list(AMINO_ACIDS),
        zmin=0,
        zmax=1,
        # Somehow makes the plot approximately the right size
        width=max(len(x_labels), 20) * 20,
        title=probabilities["original_id"].iloc[0],
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            tickmode="linear",
        ),
        yaxis=dict(
            tickmode="linear",
        ),
    )

    fig.add_trace(
        go.Scatter(x=x_labels, y=probabilities["wild_type_amino_acid"], mode="markers")
    )

    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)

    return fig


def run(**kwargs):
    """WIP Do Not Use

    mandatory:
    * probabilities_file
    """
    required_kwargs = [
        "protocol",
        "prefix",
        "stage_name",
        "probabilities_file",
    ]
    check_required(kwargs, required_kwargs)
    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager()
    stage = file_manager.create_stage(
        result_kwargs["prefix"], result_kwargs["stage_name"]
    )

    probabilities_all = pandas.read_csv(result_kwargs["probabilities_file"])
    assert (
        list(probabilities_all.columns) == PROBABILITIES_COLUMNS
    ), f"probabilities file is expected to have the following columns: {PROBABILITIES_COLUMNS}"
    number_of_proteins = len(set(probabilities_all["id"]))

    for sequence_id, probabilities in tqdm(
        probabilities_all.groupby("id"), total=number_of_proteins
    ):
        fig = plot(probabilities)
        plotly.offline.plot(
            fig, filename=str(Path(stage).joinpath(f"{sequence_id}.html"))
        )

    return result_kwargs

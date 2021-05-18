import plotly
import plotly.express as px
from plotly.graph_objects import Figure as _Figure
from pandas import DataFrame


def render_3D_scatter_plotly(embeddings_dataframe: DataFrame) -> _Figure:
    """
    Return a Plotly Figure (3D scatter plot) based on a DataFrame containing three components.

    :param embeddings_dataframe: the DataFrame *must* contain three numerical columns called `component_0`,
            `component_1` and `component_2`. The DataFrame index will be used to identify the points in the
            scatter plot. Optionally, the DataFrame may contain a column called `label` which will be used
            to color the points in the scatter plot.
    :return: A 3D scatter plot
    """

    if 'label' in embeddings_dataframe.columns:
        fig = px.scatter_3d(embeddings_dataframe,
                            x='component_0',
                            y='component_1',
                            z='component_2',
                            color='label',
                            symbol='label',
                            hover_name=embeddings_dataframe.index,
                            hover_data=["label"]
                            )
    else:
        fig = px.scatter_3d(embeddings_dataframe,
                            x='component_0',
                            y='component_1',
                            z='component_2',
                            hover_name=embeddings_dataframe.index,
                            )

    fig.update_layout(
        # Remove axes ticks and labels as they are usually not informative
        scene=dict(
            xaxis=dict(
                showticklabels=False,
                showspikes=False,
                title=""
            ),
            yaxis=dict(
                showticklabels=False,
                showspikes=False,
                title=""
            ),
            zaxis=dict(
                showticklabels=False,
                showspikes=False,
                title=""
            )
        ),
    )

    return fig


def render_scatter_plotly(embeddings_dataframe: DataFrame) -> _Figure:
    """
    Return a Plotly Figure (2D scatter plot) based on a DataFrame containing three components.

    :param embeddings_dataframe: the DataFrame *must* contain two numerical columns called `component_0`
            and `component_1`. The DataFrame index will be used to identify the points in the
            scatter plot. Optionally, the DataFrame may contain a column called `label` which will be used
            to color the points in the scatter plot.
    :return: A 2D scatter plot
    """

    hover_data = []
    if "original_id" in embeddings_dataframe.columns:
        hover_data.append("original_id")
    if "label" in embeddings_dataframe.columns:
        hover_data.append("label")

    if 'label' in embeddings_dataframe.columns:
        fig = px.scatter(embeddings_dataframe,
                         x='component_0',
                         y='component_1',
                         color='label',
                         symbol='label',
                         hover_name=embeddings_dataframe.index,
                         hover_data=hover_data
                         )
    else:
        fig = px.scatter(embeddings_dataframe,
                         x='component_0',
                         y='component_1',
                         hover_name=embeddings_dataframe.index,
                         hover_data=hover_data
                         )

    fig.update_layout(
        # Remove axes ticks and labels as they are usually not informative
        scene=dict(
            xaxis=dict(
                showticklabels=False,
                showspikes=False,
                title=""
            ),
            yaxis=dict(
                showticklabels=False,
                showspikes=False,
                title=""
            )
        ),
    )

    return fig


def save_plotly_figure_to_html(figure: _Figure, path: str) -> None:
    """
    Store plotly figure as interactive HTML file

    :param figure: A Plotly Figure
    :param path: A string representing the path and/or filename where the HTML figure should be stored
         (e.g.: /path/to/figure.html).
    """
    plotly.offline.plot(figure, filename=path)

import plotly
import plotly.express as px
from pandas import DataFrame


def render_3D_scatter_plotly(embeddings_dataframe: DataFrame):
    if 'label' in embeddings_dataframe.columns:
        fig = px.scatter_3d(embeddings_dataframe,
                            template='ggplot2',
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
                            template='ggplot2',
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


def render_scatter_plotly(embeddings_dataframe: DataFrame):
    if 'label' in embeddings_dataframe.columns:
        fig = px.scatter(embeddings_dataframe,
                         template='ggplot2',
                         x='component_0',
                         y='component_1',
                         color='label',
                         symbol='label',
                         hover_name=embeddings_dataframe.index,
                         hover_data=["label"]
                         )
    else:
        fig = px.scatter(embeddings_dataframe,
                         template='ggplot2',
                         x='component_0',
                         y='component_1',
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
            )
        ),
    )

    return fig


def save_plotly_figure_to_html(figure, path):
    plotly.offline.plot(figure, filename=path)

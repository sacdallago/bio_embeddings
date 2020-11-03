"""Display t-SNE or UMAP projections from the project stage"""

from bio_embeddings.visualize.plotly_plots import (
    render_3D_scatter_plotly,
    render_scatter_plotly,
    save_plotly_figure_to_html,
)

__all__ = [
    "render_3D_scatter_plotly",
    "render_scatter_plotly",
    "save_plotly_figure_to_html",
]

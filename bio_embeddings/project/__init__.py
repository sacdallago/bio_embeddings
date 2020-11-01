"""Visualize high dimensional data with t-SNE or UMAP projections"""

from bio_embeddings.project.tsne import tsne_reduce
from bio_embeddings.project.umap import umap_reduce

__all__ = [
    "tsne_reduce",
    "umap_reduce"
]
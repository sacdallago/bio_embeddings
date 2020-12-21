"""Methods for predicting properties of proteins, both on a per-residue and
per-protein level, including supervised (pre-trained) and unsupervised (nearest
neighbour search) methods
"""

from bio_embeddings.extract.unsupervised_utilities import (
    pairwise_distance_matrix_from_embeddings_and_annotations,
    get_k_nearest_neighbours,
)
from bio_embeddings.extract.basic import BasicAnnotationExtractor

__all__ = [
    "BasicAnnotationExtractor",
    "get_k_nearest_neighbours",
    "pairwise_distance_matrix_from_embeddings_and_annotations",
]

import collections
import h5py
import numpy as np

from typing import List
from sklearn.metrics import pairwise_distances as _pairwise_distances


PairwiseDistanceMatrixResult = collections.namedtuple('PairwiseDistanceMatrixResult', 'pairwise_matrix queries references')


def pairwise_distance_matrix_from_embeddings_and_annotations(query_embeddings_path: str, reference_embeddings_path: str,
                                                             metric: str = "euclidean", n_jobs: int = 1) -> PairwiseDistanceMatrixResult:
    """

    :param n_jobs: int, see scikit-learn documentation
    :param metric: Metric to use (string!), see scikit-learn documentation
    :param query_embeddings_path: A string defining a path to an h5 file
    :param reference_embeddings_path: A string defining a path to an h5 file
    :return: A tuple containing:
        - pairwise_matrix: the pairwise distances between queries and references
        - queries: A list of strings defining the queries
        - references: A list of strings defining the references
    """
    references: List[str]
    queries: List[str]
    reference_embeddings = list()
    query_embeddings = list()

    with h5py.File(reference_embeddings_path, 'r') as reference_embeddings_file,\
          h5py.File(query_embeddings_path, 'r') as query_embeddings_file:

        references = list(reference_embeddings_file.keys())
        queries = list(query_embeddings_file.keys())

        for refereince_identifier in references:
            reference_embeddings.append(np.array(reference_embeddings_file[refereince_identifier]))

        for query_identifier in queries:
            query_embeddings.append(np.array(query_embeddings_file[query_identifier]))

    pairwise_distances = _pairwise_distances(
        query_embeddings,
        reference_embeddings,
        metric=metric,
        n_jobs=n_jobs
    )

    return PairwiseDistanceMatrixResult(pairwise_matrix=pairwise_distances, queries=queries, references=references)


def get_k_nearest_neighbours(pairwise_matrix: np.array, k: int = 1) -> (List[int], np.array):
    """

    :param pairwise_matrix: an np.array with columns as queries and rows as targets
    :param k: the number of k-nn's to return
    :return: a list of tuples with indices of the nearest neighbour and distance to them (sorted by distance asc.)
    """
    resulting_indices = list()
    resulting_distances = list()

    for i, neighbour_distances in enumerate(pairwise_matrix):
        nearest_neighbour_indices = np.argpartition(neighbour_distances, k)[:k]
        nearest_neighbour_distances = np.array(list(map(neighbour_distances.__getitem__, nearest_neighbour_indices)))

        # nearest_neighbours will appear in an arbitrary order.
        # We want to ensure that the distances and indices are sorted by ascending distance
        # The following code shuffles both lists around to make sure that indices and distances are sorted equally
        nearest_neighbour_distances, nearest_neighbour_indices = (list(t) for t in zip(
            *sorted(
                zip(nearest_neighbour_distances, nearest_neighbour_indices)
            )
        ))

        resulting_indices.append(nearest_neighbour_indices)
        resulting_distances.append(nearest_neighbour_distances)

    return resulting_indices, np.array(resulting_distances)

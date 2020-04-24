from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE


def tsne_reduce(embeddings, **kwargs):
    tsne_params = dict()
    pairwise_distances_params = dict()

    pairwise_distances_params['metric'] = kwargs.get('metric', 'cosine')
    pairwise_distances_params['n_jobs'] = kwargs.get('n_jobs', -1)

    tsne_params['n_components'] = kwargs.get('n_components', 3)
    tsne_params['perplexity'] = kwargs.get('perplexity', 6)
    tsne_params['random_state'] = kwargs.get('random_state', 420)
    tsne_params['n_iter'] = kwargs.get('n_iter', 15000)
    tsne_params['verbose'] = kwargs.get('verbose', 1)

    # Important: set precomputed as metric for tsne
    tsne_params['metric'] = 'precomputed'

    distance_matrix = pairwise_distances(embeddings, embeddings, **pairwise_distances_params)
    transformed_embeddings = TSNE(**tsne_params).fit_transform(distance_matrix)

    return transformed_embeddings

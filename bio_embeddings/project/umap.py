from umap import UMAP


def umap_reduce(embeddings, **kwargs):
    """Wrapper around :meth:`umap.UMAP` with defaults for bio_embeddings"""
    umap_params = dict()

    umap_params['n_components'] = kwargs.get('n_components', 3)
    umap_params['min_dist'] = kwargs.get('min_dist', .6)
    umap_params['spread'] = kwargs.get('spread', 1)
    umap_params['random_state'] = kwargs.get('random_state', 420)
    umap_params['n_neighbors'] = kwargs.get('n_neighbors', 15)
    umap_params['verbose'] = kwargs.get('verbose', 1)
    umap_params['metric'] = kwargs.get('metric', 'cosine')

    transformed_embeddings = UMAP(**umap_params).fit_transform(embeddings)

    return transformed_embeddings
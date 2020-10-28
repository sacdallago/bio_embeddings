from sklearn.manifold import TSNE


def tsne_reduce(embeddings, **kwargs):
    """Wrapper around :meth:`sklearn.manifold.TSNE` with defaults for bio_embeddings"""
    tsne_params = dict()

    tsne_params['n_components'] = kwargs.get('n_components', 3)
    tsne_params['perplexity'] = kwargs.get('perplexity', 6)
    tsne_params['random_state'] = kwargs.get('random_state', 420)
    tsne_params['n_iter'] = kwargs.get('n_iter', 15000)
    tsne_params['verbose'] = kwargs.get('verbose', 1)
    tsne_params['n_jobs'] = kwargs.get('n_jobs', -1)
    tsne_params['metric'] = kwargs.get('metric', 'cosine')

    transformed_embeddings = TSNE(**tsne_params).fit_transform(embeddings)

    return transformed_embeddings

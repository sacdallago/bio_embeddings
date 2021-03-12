from numpy import ndarray

from bio_embeddings.embed import (
    SeqVecEmbedder,
    UniRepEmbedder,
    CPCProtEmbedder,
    EmbedderInterface,
)


def check_embedding(embedder: EmbedderInterface, embedding, sequence: str):
    """Checks that the shape of the embeddings looks credible"""
    assert isinstance(embedding, ndarray)
    if embedder.__class__ == SeqVecEmbedder:
        assert embedding.shape[1] == len(sequence)
    elif embedder.__class__ == UniRepEmbedder:
        # See https://github.com/ElArkk/jax-unirep/issues/85
        assert embedding.shape[0] == len(sequence) + 1
    elif embedder.__class__ == CPCProtEmbedder:
        # There is only a per-protein embedding for CPCProt
        assert embedding.shape == (512,)
    else:
        assert embedding.shape[0] == len(sequence)

    # Check reduce_per_protein
    # https://github.com/sacdallago/bio_embeddings/issues/85
    assert embedder.reduce_per_protein(embedding).shape == (
        embedder.embedding_dimension,
    )

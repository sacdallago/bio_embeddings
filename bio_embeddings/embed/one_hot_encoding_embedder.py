import re

import numpy
from numpy import ndarray

from bio_embeddings.embed import EmbedderInterface

AMINO_ACIDS = numpy.asarray(list("ACDEFGHIKLMNPQRSTVWXY"))


class OneHotEncodingEmbedder(EmbedderInterface):
    """Baseline embedder: One hot encoding as per-residue embedding, amino acid composition for per-protein

    This embedder is meant to be used as naive baseline for comparing different types of inputs or training method.

    While option such as device aren't used, you may still pass them for consistency.
    """

    number_of_layers = 1
    embedding_dimension = len(AMINO_ACIDS)
    name = "one_hot_encoding"

    def embed(self, sequence: str) -> ndarray:
        if not sequence:
            return numpy.zeros((len(AMINO_ACIDS), 0))
        sequence = re.sub(r"[UZOB]", "X", sequence)
        one_hot = [AMINO_ACIDS == i for i in sequence]
        return numpy.stack(one_hot).astype(numpy.float32)

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        """This returns the amino acid composition of the sequence as vector"""
        return embedding.mean(axis=0)

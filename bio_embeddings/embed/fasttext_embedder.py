import re

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from numpy import ndarray

from bio_embeddings.embed.embedder_interfaces import EmbedderInterface


class FastTextEmbedder(EmbedderInterface):
    name = "fasttext"
    embedding_dimension = 512
    number_of_layers = 1
    necessary_files = ["model_file"]

    def __init__(self, **kwargs):
        """

        :param model_file: path of model file. If not supplied, will be downloaded.
        """
        super().__init__(**kwargs)

        self._model_file = self._options.get("model_file")

        self._model = KeyedVectors.load_word2vec_format(self._model_file, binary=False)
        self._vector_size = 512
        self._zero_vector = np.zeros(self._vector_size, dtype=np.float32)
        self._window_size = 3

    def embed(self, sequence: str) -> ndarray:
        sequence = re.sub(r"[UZOB]", "X", sequence)
        # pad sequence with special character (only 3-mers are considered)
        padded_sequence = "-" + sequence + "-"

        # container
        embedding = np.zeros((len(sequence), self._vector_size), dtype=np.float32)

        # for each aa in the sequence, retrieve k-mer
        for index in range(len(padded_sequence)):
            try:
                k_mer = "".join(padded_sequence[index : index + self._window_size])
                embedding[index, :] = self._get_kmer_representation(k_mer)
            # end of sequence reached
            except IndexError:
                return embedding

    def _get_kmer_representation(self, k_mer):
        # try to retrieve embedding for k-mer
        try:
            return self._model[k_mer]
        # in case of padded or out-of-vocab character
        except KeyError:
            # if single AA was not part of corpus (or no AA)
            if len(k_mer) <= 1:
                return self._zero_vector
            # handle border cases at start/end of seq
            elif "-" in k_mer:
                idx_center = int(len(k_mer) / 2)
                return self._get_kmer_representation(k_mer[idx_center])

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        return embedding.mean(axis=0)

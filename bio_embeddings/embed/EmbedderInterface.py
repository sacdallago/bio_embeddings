"""
Abstract interface for Embedder.

Authors:
  Christian Dallago
"""

import abc
from typing import List

from bio_embeddings.utilities.exceptions import NoEmbeddingException


class EmbedderInterface(object, metaclass=abc.ABCMeta):

    def __init__(self):
        """
        Initializer accepts location of a pre-trained model and options
        """
        self._options = None

        pass

    @abc.abstractmethod
    def embed(self, sequence: str):
        """
        Returns embedding for one sequence.

        :param sequence: Valid amino acid sequence as String
        :return: An embedding of the sequence.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def embed_many(self, sequences: List[str]) -> List[List[List[float]]]:
        """
        Returns embedding for one sequence.

        :param sequences: List of proteins as AA strings
        :return: A list object with embeddings of the sequences.
        """

        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def reduce_per_protein(embedding):
        """
        For a variable size embedding, returns a fixed size embedding encoding all information of a sequence.

        :param embedding: the embedding
        :return: A fixed size embedding (a vector of size N, where N is fixed)
        """

        raise NotImplementedError

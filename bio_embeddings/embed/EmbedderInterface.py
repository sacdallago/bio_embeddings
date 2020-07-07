"""
Abstract interface for Embedder.

Authors:
  Christian Dallago
"""

import abc
from typing import List, Generator, Optional

from numpy import ndarray


class EmbedderInterface(object, metaclass=abc.ABCMeta):
    def __init__(self):
        """
        Initializer accepts location of a pre-trained model and options
        """
        self._options = None

    @abc.abstractmethod
    def embed(self, sequence: str) -> ndarray:
        """
        Returns embedding for one sequence.

        :param sequence: Valid amino acid sequence as String
        :return: An embedding of the sequence.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def embed_many(self, sequences: List[str], batch_size: Optional[int] = None) -> Generator[ndarray, None, None]:
        """
        Returns embedding for one sequence.

        :param sequences: List of proteins as AA strings
        :param batch_size: For embedders that profit from batching, this is maximum number of AA per batch
        :return: A list object with embeddings of the sequences.
        """

        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        """
        For a variable size embedding, returns a fixed size embedding encoding all information of a sequence.

        :param embedding: the embedding
        :return: A fixed size embedding (a vector of size N, where N is fixed)
        """

        raise NotImplementedError

"""
Abstract interface for Embedder.

Authors:
  Christian Dallago
"""

import abc

from bio_embeddings.utilities.exceptions import NoEmbeddingException


class EmbedderInterface(object, metaclass=abc.ABCMeta):

    def __init__(self):
        """
        Initializer accepts location of a pre-trained model and options
        """
        self._options = None
        self._embedding = None
        self._sequence = None

        pass

    @abc.abstractmethod
    def embed(self, sequence):
        """
        Returns embedding for one sequence.

        :param sequence: Valid amino acid sequence as String
        :return: An embedding of the sequence.
        """

        raise NotImplementedError

    def get_embedding(self):
        """
        Returns the embedding
        :return: An seqvec embedding or NoEmbeddingException
        """

        if self._embedding is None:
            raise NoEmbeddingException

        return self._embedding

    def get_sequence(self):
        """
        :return: A string representing the sequence which was embedded. None if no embedding has been calculated
        """

        return self._sequence

    @abc.abstractmethod
    def get_features(self, embedding=None):
        """
        Returns a FeaturesCollection object. Embedding must not be None, otherwise rises NoEmbeddingException

        :param embedding: OPTIONAL embedding from which to calculate the features. Useful when running on parallel systems
        :return: A bag with various AA-specific and global features
        """
        raise NotImplementedError

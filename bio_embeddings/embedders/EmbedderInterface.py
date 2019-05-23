"""
Abstract interface for Embedder.

Authors:
  Christian Dallago
"""

import abc


class EmbedderInterface(object, metaclass=abc.ABCMeta):

    def __init__(self):
        """
        Initializer accepts location of a pre-trained model and options
        """
        self._options = None
        self._embedding = None
        self._sequence = None
        self._model = None

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
        :return: An elmo embedding or NoEmbeddingException
        """

        if self._embedding is None:
            raise NoEmbeddingException

        return self._embedding

    @abc.abstractmethod
    def get_features(sel):
        """
        Returns a bag with Objects of type Feature. Embedding must not be None, otherwise rises NoEmbeddingException
        :return: A bag with various AA-specific and global features
        """
        raise NotImplementedError


class NoEmbeddingException(Exception):
    """
    Exception to handle the case no embedding has been computed, and one requests features.
    """

"""
Abstract interface for Embedder.

Authors:
  Christian Dallago
"""

import abc


class EmbedderInterface(object, metaclass=abc.ABCMeta):

    def __init__(self, weights_file, options_file):
        """
        Initializer accepts location of a pre-trained model and options

        :param weights_file: location of the weights
        :param options_file: location of the model options, if any
        """
        self._weights_file = weights_file
        self._options_file = options_file
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

        # TODO: Test that sequence is a valid sequence

        self._sequence = sequence
        raise NotImplementedError

    @abc.abstractmethod
    def get_features(self):
        """
        Returns a bag with Objects of type Feature. Embedding must not be None, otherwise rises NoEmbeddingException
        :return: A bag with various AA-specific and global features
        """
        raise NotImplementedError


class NoEmbeddingException(Exception):
    """
    Exception to handle the case no embedding has been computed, and one requests features.
    """

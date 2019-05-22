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
        self.weights_file = weights_file
        self.options_file = options_file

        pass

    @abc.abstractmethod
    def get_embedding(self, sequence):
        """
        Returns embedding for one sequence.

        :param sequence: Valid amino acid sequence as String
        :return: An embedding of the sequence.
        """
        raise NotImplementedError
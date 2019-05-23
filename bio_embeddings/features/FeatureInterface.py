"""
Abstract interface for a Feature.

Authors:
  Christian Dallago
"""

import abc


class FeatureInterface(object, metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    @abc.abstractmethod
    def isAAFeature(self):
        """
        Returns if feature is Amino Acid specific (or global, e.g. protein-wide)
        :return: Bool
        """
        raise NotImplementedError

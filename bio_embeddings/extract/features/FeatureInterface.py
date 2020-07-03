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


class InvalidFeatureException(Exception):
    """
    Exception to handle most of regex or string matching exceptions. E.g. DSSP8 dictionary or protein location classes
    """

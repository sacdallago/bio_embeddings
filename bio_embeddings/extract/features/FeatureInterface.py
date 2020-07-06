"""
Abstract interface for a Feature.

Authors:
  Christian Dallago
"""

import abc
from enum import Enum


class FeatureInterface(Enum, metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def isAAFeature():
        """
        Returns if feature is Amino Acid specific (or global, e.g. protein-wide)
        :return: Bool
        """
        raise NotImplementedError

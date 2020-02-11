"""
Abstract interface for Embedder.

Authors:
  Christian Dallago
"""

import abc
from bio_embeddings.extract_features.features import FeaturesCollection


class FeatureExtractorInterface(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_features(self, embedding=None) -> FeaturesCollection:
        """
        Returns a FeaturesCollection object. Embedding must not be None, otherwise rises NoEmbeddingException

        :param embedding: embedding from which to calculate the features.
        :return: A bag with various AA-specific and global features
        """
        raise NotImplementedError

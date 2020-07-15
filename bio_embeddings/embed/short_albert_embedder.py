import logging

import numpy as np
from numpy import ndarray

from bio_embeddings.embed.albert_embedder import AlbertEmbedder
from bio_embeddings.utilities import (
    SequenceTooLongException,
)

logger = logging.getLogger(__name__)


class ShortAlbertEmbedder(AlbertEmbedder):
    def __init__(self, **kwargs):
        """
        Initialize Short Albert embedder. This will extend the standard Albert embedder, but additionally throw consider length limitations

        :param ignore_long_proteins: True will ignore proteins longer than 510. False: will throw exception if embedding sequence with length > 510. Default: True
        """
        super().__init__(**kwargs)

        self._ignore_long_proteins = self._options.get('ignore_long_proteins', True)
        self._max_sequence_length = 510

    def embed(self, sequence: str) -> ndarray:
        sequence_length = len(sequence)
        if sequence_length > self._max_sequence_length:
            if not self._ignore_long_proteins:
                raise SequenceTooLongException()
            else:
                logger.info('''Trying to embed a sequence of length {}, but maximal length allowed is {}.
                            The embedding for this sequence will be zeroes!'''.format(
                    sequence_length, self._max_sequence_length
                ))
                return np.zeros((sequence_length, 4096))

        return super(ShortAlbertEmbedder, self).embed(sequence)

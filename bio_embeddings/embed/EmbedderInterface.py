"""
Abstract interface for Embedder.

Authors:
  Christian Dallago
"""

import abc
import logging
from typing import List, Generator, Optional, Iterable

from numpy import ndarray

logger = logging.getLogger(__name__)


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

    def embed_batch(self, batch: List[str]) -> Generator[ndarray, None, None]:
        """ Computes the embeddings from all sequences in the batch

        The provided implementation is dummy implementation that should be
        overwritten with the appropriate batching method for the model. """
        for sequence in batch:
            yield self.embed(sequence)

    def embed_many(
        self, sequences: Iterable[str], batch_size: Optional[int] = None
    ) -> Generator[ndarray, None, None]:
        """
        Returns embedding for one sequence.

        :param sequences: List of proteins as AA strings
        :param batch_size: For embedders that profit from batching, this is maximum number of AA per batch
        :return: A list object with embeddings of the sequences.
        """

        if batch_size:
            batch = []
            length = 0
            for sequence in sequences:
                if len(sequence) > batch_size:
                    logger.warning(
                        f"A sequence is {len(sequence)} residues long, "
                        f"which is longer than your `batch_size` parameter which is {batch_size}"
                    )
                    yield from self.embed_batch([sequence])
                    continue
                if length + len(sequence) >= batch_size:
                    yield from self.embed_batch(batch)
                    batch = []
                    length = 0
                batch.append(sequence)
                length += len(sequence)
            yield from self.embed_batch(batch)
        else:
            for seq in sequences:
                yield self.embed(seq)

    @staticmethod
    @abc.abstractmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        """
        For a variable size embedding, returns a fixed size embedding encoding all information of a sequence.

        :param embedding: the embedding
        :return: A fixed size embedding (a vector of size N, where N is fixed)
        """

        raise NotImplementedError

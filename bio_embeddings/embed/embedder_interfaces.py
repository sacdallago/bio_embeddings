"""
Abstract interface for Embedder.

Authors:
  Christian Dallago
"""

import abc
import logging
from typing import List, Generator, Optional, Iterable, ClassVar, Any, Dict
from typing import Type, TypeVar

import torch

from numpy import ndarray

# https://stackoverflow.com/a/39205612/3549270
EmbedderInterfaceSubclass = TypeVar(
    "EmbedderInterfaceSubclass", bound="EmbedderInterface"
)

logger = logging.getLogger(__name__)


class EmbedderInterface(abc.ABC):
    name: ClassVar[str]
    # An integer representing the size of the embedding.
    embedding_dimension: ClassVar[int]
    # An integer representing the number of layers from the RAW output of the LM.
    number_of_layers: ClassVar[int]
    _use_cpu: bool
    _device: torch.device
    _options: Dict[str, Any]

    def __init__(self, use_cpu: bool = False, **kwargs):
        """
        Initializer accepts location of a pre-trained model and options
        """
        self._options = kwargs
        self._use_cpu = use_cpu
        self._device = torch.device(
            "cuda:0" if torch.cuda.is_available() and not self._use_cpu else "cpu"
        )

    @classmethod
    @abc.abstractmethod
    def with_download(
        cls: Type[EmbedderInterfaceSubclass], **kwargs
    ) -> EmbedderInterfaceSubclass:
        """ Convenience function to create an instance after downloading files. """
        raise NotImplementedError

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


class EmbedderWithFallback(EmbedderInterface, abc.ABC):
    """ Batching embedder that will fallback to the CPU if the embedding on the GPU failed """
    _model: Any

    @abc.abstractmethod
    def _embed_batch_impl(
        self, batch: List[str], model: Any
    ) -> Generator[ndarray, None, None]:
        ...

    @abc.abstractmethod
    def _get_fallback_model(self):
        """ Returns a (cached) cpu model """
        ...

    def embed_batch(self, batch: List[str]) -> Generator[ndarray, None, None]:
        """ Tries to get the embeddings in this order:
          * Full batch GPU
          * Single Sequence GPU
          * Single Sequence CPU

        Single sequence processing is done in case of runtime error due to
        a) very long sequence or b) too large batch size
        If this fails, you might want to consider lowering batch_size and/or
        cutting very long sequences into smaller chunks

        Returns unprocessed embeddings
        """
        # No point in having a fallback model when the normal model is CPU already
        if self._use_cpu:
            yield from self._embed_batch_impl(batch, self._model)
            return

        try:
            yield from self._embed_batch_impl(batch, self._model)
        except RuntimeError as e:
            if len(batch) == 1:
                logger.error(
                    f"RuntimeError for sequence with {len(batch[0])} residues: {e}. "
                    f"This most likely means that you don't have enough GPU RAM to embed a protein this long. "
                    f"Embedding on the CPU instead, which is very slow"
                )
                yield from self._embed_batch_impl(batch, self._get_fallback_model())
            else:
                logger.error(
                    f"Error processing batch of {len(batch)} sequences: {e}. "
                    f"You might want to consider adjusting the `batch_size` parameter. "
                    f"Will try to embed each sequence in the set individually on the GPU."
                )
                for sequence in batch:
                    try:
                        yield from self._embed_batch_impl([sequence], self._model)
                    except RuntimeError as e:
                        logger.error(
                            f"RuntimeError for sequence with {len(sequence)} residues: {e}. "
                            f"This most likely means that you don't have enough GPU RAM to embed a protein this long."
                        )
                        yield from self._embed_batch_impl(
                            [sequence], self._get_fallback_model()
                        )

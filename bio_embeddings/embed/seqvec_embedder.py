import logging
import tempfile
from typing import List, Optional, Generator

import torch
from allennlp.commands.elmo import ElmoEmbedder
from numpy import ndarray

from bio_embeddings.embed.embedder_interface import EmbedderInterface
from bio_embeddings.utilities import get_model_file

logger = logging.getLogger(__name__)


class SeqVecEmbedder(EmbedderInterface):
    name = "seqvec"

    _weights_file: str
    _options_file: str
    _use_cpu: bool
    _elmo_model: ElmoEmbedder
    # The fallback model running on the cpu, which will be initialized if needed
    _elmo_model_fallback: Optional[ElmoEmbedder] = None

    def __init__(self, **kwargs):
        """
        Initialize Elmo embedder. Can define non-positional arguments for paths of files and other settings.

        :param weights_file: path of weights file
        :param options_file: path of options file
        :param use_cpu: overwrite autodiscovery and force CPU use
        :param max_amino_acids: max # of amino acids to include in embed_many batches. Default: 15k AA
        """
        super().__init__()

        self._options = kwargs

        # Get file locations from kwargs
        self._weights_file = self._options["weights_file"]
        self._options_file = self._options["options_file"]
        self._use_cpu = self._options.get("use_cpu", False)

        if torch.cuda.is_available() and not self._use_cpu:
            logger.info("CUDA available, using the GPU")

            # Set CUDA device for ELMO machine
            cuda_device = 0
        else:
            logger.info("CUDA NOT available, using the CPU. This is slow")

            # Set CUDA device for ELMO machine
            cuda_device = -1

        self._elmo_model = ElmoEmbedder(
            weight_file=self._weights_file,
            options_file=self._options_file,
            cuda_device=cuda_device,
        )

    @classmethod
    def with_download(cls, **kwargs):
        necessary_files = ['weights_file', 'options_file']

        if kwargs.get('seqvec_version') == 2 or kwargs.get('vocabulary_file'):
            necessary_files.append('vocabulary_file')
            kwargs['seqvec_version'] = 2

        for file in necessary_files:
            if not kwargs.get(file):
                f = tempfile.NamedTemporaryFile()

                get_model_file(path=f.name, model=cls.name.format(str(kwargs.get('seqvec_version', 1))), file=file)

                kwargs[file] = f.name
        return cls(**kwargs)

    def embed(self, sequence: str) -> ndarray:
        return self._elmo_model.embed_sentence(list(sequence))

    def embed_fallback(self, sequence: str) -> ndarray:
        if not self._elmo_model_fallback:
            logger.warning(
                "Loading model for CPU into RAM. Embedding on the CPU is very slow and you should avoid it."
            )
            self._elmo_model_fallback = ElmoEmbedder(
                weight_file=self._weights_file,
                options_file=self._options_file,
                cuda_device=-1,
            )
        return self._elmo_model_fallback.embed_sentence(list(sequence))

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
        # elmo expect a `List[str]` as it was meant for tokens/words with more than one character.
        try:
            yield from self._elmo_model.embed_batch([list(seq) for seq in batch])
        except RuntimeError as e:
            if len(batch) == 1:
                logger.error(
                    f"RuntimeError for sequence with {len(batch[0])} residues: {e}. "
                    f"This most likely means that you don't have enough GPU RAM to embed a protein this long. "
                    f"Embedding on the CPU instead, which is very slow"
                )
                yield self.embed_fallback(batch[0])
            else:
                logger.error(
                    f"Error processing batch of {len(batch)} sequences: {e}. "
                    f"You might want to consider adjusting the `batch_size` parameter. "
                    f"Will try to embed each sequence in the set individually on the GPU."
                )
                for seq in batch:
                    try:
                        yield self._elmo_model.embed_sentence(list(seq))
                    except RuntimeError as e:
                        logger.error(
                            f"RuntimeError for sequence with {len(seq)} residues: {e}. "
                            f"This most likely means that you don't have enough GPU RAM to embed a protein this long."
                        )
                        yield self.embed_fallback(seq)

    @staticmethod
    def reduce_per_protein(embedding):
        return embedding.sum(0).mean(0)

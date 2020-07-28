import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Generator

import torch
from allennlp.commands.elmo import ElmoEmbedder
from numpy import ndarray

from bio_embeddings.embed.embedder_interfaces import EmbedderWithFallback
from bio_embeddings.utilities import get_model_file

logger = logging.getLogger(__name__)


class SeqVecEmbedder(EmbedderWithFallback):
    name = "seqvec"
    embedding_dimension = 1024
    number_of_layers = 3

    _weights_file: str
    _options_file: str
    model: ElmoEmbedder
    # The fallback model running on the cpu, which will be initialized if needed
    model_fallback: Optional[ElmoEmbedder] = None

    def __init__(self, **kwargs):
        """
        Initialize Elmo embedder. Can define non-positional arguments for paths of files and other settings.

        :param weights_file: path of weights file
        :param options_file: path of options file
        :param model_directory: Alternative of weights_file/options_file
        :param use_cpu: overwrite autodiscovery and force CPU use
        :param max_amino_acids: max # of amino acids to include in embed_many batches. Default: 15k AA
        """
        super().__init__(**kwargs)

        # Get file locations from kwargs
        if "model_directory" in self._options:
            self._weights_file = str(
                Path(self._options["model_directory"]).joinpath("weights_file")
            )
            self._options_file = str(
                Path(self._options["model_directory"]).joinpath("options_file")
            )
        else:
            self._weights_file = self._options["weights_file"]
            self._options_file = self._options["options_file"]

        # TODO: Only SeqVec is checking for cuda availibilty, this should be done by all or none instead
        # Additionally, this should use self.device.index
        if torch.cuda.is_available() and not self._use_cpu:
            logger.info("CUDA available, using the GPU")
            cuda_device = 0
        else:
            logger.info("CUDA NOT available, using the CPU. This is slow")
            cuda_device = -1

        self.model = ElmoEmbedder(
            weight_file=self._weights_file,
            options_file=self._options_file,
            cuda_device=cuda_device,
        )

    @classmethod
    def with_download(cls, **kwargs) -> "SeqVecEmbedder":
        necessary_files = ["weights_file", "options_file"]

        keep_tempfiles_alive = []
        for file in necessary_files:
            if not kwargs.get(file):
                f = tempfile.NamedTemporaryFile()
                keep_tempfiles_alive.append(f)

                get_model_file(path=f.name, model=cls.name, file=file)

                kwargs[file] = f.name
        return cls(**kwargs)

    def embed(self, sequence: str) -> ndarray:
        return self.model.embed_sentence(list(sequence))

    def _get_fallback_model(self) -> ElmoEmbedder:
        if not self.model_fallback:
            logger.warning(
                "Loading model for CPU into RAM. Embedding on the CPU is very slow and you should avoid it."
            )
            self.model_fallback = ElmoEmbedder(
                weight_file=self._weights_file,
                options_file=self._options_file,
                cuda_device=-1,
            )
        return self.model_fallback

    def _embed_batch_impl(
        self, batch: List[str], model: ElmoEmbedder
    ) -> Generator[ndarray, None, None]:
        # elmo expect a `List[str]` as it was meant for tokens/words with more than one character.
        yield from model.embed_batch([list(seq) for seq in batch])

    @staticmethod
    def reduce_per_protein(embedding):
        return embedding.sum(0).mean(0)

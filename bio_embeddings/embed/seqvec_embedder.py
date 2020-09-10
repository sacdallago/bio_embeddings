import logging
from pathlib import Path
from typing import List, Optional, Generator

from allennlp.commands.elmo import ElmoEmbedder
from numpy import ndarray

from bio_embeddings.embed.embedder_interfaces import EmbedderWithFallback

logger = logging.getLogger(__name__)


class SeqVecEmbedder(EmbedderWithFallback):
    name = "seqvec"
    embedding_dimension = 1024
    number_of_layers = 3

    _weights_file: str
    _options_file: str
    _model: ElmoEmbedder
    # The fallback model running on the cpu, which will be initialized if needed
    _model_fallback: Optional[ElmoEmbedder] = None
    _necessary_files = ["weights_file", "options_file"]

    def __init__(self, **kwargs):
        """
        Initialize Elmo embedder. Can define non-positional arguments for paths of files and other settings.

        :param weights_file: path of weights file
        :param options_file: path of options file
        :param model_directory: Alternative of weights_file/options_file
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

        if self._device.type == "cuda":
            logger.info("CUDA available, using the GPU")
            cuda_device = self._device.index or 0
        else:
            logger.info("CUDA NOT available, using the CPU. This is slow")
            cuda_device = -1

        self._model = ElmoEmbedder(
            weight_file=self._weights_file,
            options_file=self._options_file,
            cuda_device=cuda_device,
        )

    def embed(self, sequence: str) -> ndarray:
        return self._model.embed_sentence(list(sequence))

    def _get_fallback_model(self) -> ElmoEmbedder:
        if not self._model_fallback:
            logger.warning(
                "Loading model for CPU into RAM. Embedding on the CPU is very slow and you should avoid it."
            )
            self._model_fallback = ElmoEmbedder(
                weight_file=self._weights_file,
                options_file=self._options_file,
                cuda_device=-1,
            )
        return self._model_fallback

    def _embed_batch_impl(
        self, batch: List[str], model: ElmoEmbedder
    ) -> Generator[ndarray, None, None]:
        # elmo expect a `List[str]` as it was meant for tokens/words with more than one character.
        yield from model.embed_batch([list(seq) for seq in batch])

    @staticmethod
    def reduce_per_protein(embedding):
        return embedding.sum(0).mean(0)

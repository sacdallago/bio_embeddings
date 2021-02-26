import logging
from pathlib import Path
from typing import List, Optional, Generator

from allennlp.commands.elmo import ElmoEmbedder
from numpy import ndarray

from bio_embeddings.embed.embedder_interfaces import EmbedderWithFallback

logger = logging.getLogger(__name__)

# A random short sequence (T0922 from CASP2)
_warmup_seq = "MGSSHHHHHHSSGLVPRGSHMASVQKFPGDANCDGIVDISDAVLIMQTMANPSKYQMTDKGRINADVTGNSDGVTVLDAQFIQSYCLGLVELPPVE"


class SeqVecEmbedder(EmbedderWithFallback):
    """SeqVec Embedder

    Heinzinger, Michael, et al. "Modeling aspects of the language of life
    through transfer-learning protein sequences." BMC bioinformatics 20.1 (2019): 723.
    https://doi.org/10.1186/s12859-019-3220-8
    """
    name = "seqvec"
    embedding_dimension = 1024
    number_of_layers = 3

    _weights_file: str
    _options_file: str
    _model: ElmoEmbedder
    # The fallback model running on the cpu, which will be initialized if needed
    _model_fallback: Optional[ElmoEmbedder] = None
    necessary_files = ["weights_file", "options_file"]

    def __init__(self, warmup_rounds: int = 4, **kwargs):
        """
        Initialize Elmo embedder. Can define non-positional arguments for paths of files and other settings.

        :param warmup_rounds: A sample sequence will be embedded this often to
            work around elmo's non-determinism (https://github.com/allenai/allennlp/blob/v0.9.0/tutorials/how_to/elmo.md#notes-on-statefulness-and-non-determinism)
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

        self.warmup_rounds = warmup_rounds
        if self.warmup_rounds > 0:
            logger.info("Running ELMo warmup")
            for _ in range(self.warmup_rounds):
                self.embed(_warmup_seq)

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
            if self.warmup_rounds > 0:
                logger.info("Running CPU ELMo warmup")
                for _ in range(self.warmup_rounds):
                    self._model_fallback.embed_sentence(list(_warmup_seq))
        return self._model_fallback

    def _embed_batch_impl(
        self, batch: List[str], model: ElmoEmbedder
    ) -> Generator[ndarray, None, None]:
        # elmo expect a `List[str]` as it was meant for tokens/words with more than one character.
        yield from model.embed_batch([list(seq) for seq in batch])

    @staticmethod
    def reduce_per_protein(embedding):
        return embedding.sum(0).mean(0)

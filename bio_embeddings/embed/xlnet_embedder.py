import logging
import re
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Generator

import torch
from numpy import ndarray
# TODO: didn't have transformers installed when writing this, so no idea if XLNetConfig can be imported as such
from transformers import XLNetModel, XLNetTokenizer, XLNetConfig

from bio_embeddings.embed.embedder_interface import EmbedderInterface
from bio_embeddings.utilities import (
    SequenceEmbeddingLengthMismatchException, get_model_directories_from_zip,
)

logger = logging.getLogger(__name__)


class XLNetEmbedder(EmbedderInterface):
    name = "xlnet"

    def __init__(self, **kwargs):
        """
        Initialize XLNet embedder.

        :param model_directory:
        :param use_cpu: overwrite autodiscovery and force CPU use
        """
        super().__init__()

        self._options = kwargs

        # Get file locations from kwargs
        self._model_directory = self._options.get('model_directory')
        self._use_cpu = self._options.get('use_cpu', False)

        # utils
        self._device = torch.device(
            "cuda:0" if torch.cuda.is_available() and not self._use_cpu else "cpu"
        )

        # make model
        config = XLNetConfig.from_json_file(str(Path(self._model_directory) / 'config.json'))
        # MH: for some reason this has to be set manually. Otherwise, AssertionError during model loading
        config.vocab_size = 37

        self._model = XLNetModel.from_pretrained(str(Path(self._model_directory) / 'model.ckpt-847000'),
                                                 from_tf=True,
                                                 config=config)
        self._model = self._model.eval()
        self._model = self._model.to(self._device)
        self._tokenizer = XLNetTokenizer(str(Path(self._model_directory) / 'spm_model.model'), do_lower_case=False)

    @classmethod
    def with_download(cls, **kwargs):
        necessary_directories = ['model_directory']

        for directory in necessary_directories:
            if not kwargs.get(directory):
                f = tempfile.mkdtemp()

                get_model_directories_from_zip(path=f, model=cls.name, directory=directory)

                kwargs[directory] = f
        return cls(**kwargs)

    def embed(self, sequence: str) -> ndarray:
        sequence_length = len(sequence)
        sequence = re.sub(r"[UZOBX]", "<unk>", sequence)

        # Tokenize sequence with spaces
        sequence = " ".join(list(sequence))

        # tokenize sequence
        tokenized_sequence = torch.tensor([self._tokenizer.encode(sequence, add_special_tokens=True)]).to(self._device)

        with torch.no_grad():
            # TODO: Konstantin, you might want to have a look at this!
            try:
                # drop batch dimension
                embedding = self._model(tokenized_sequence)[0].squeeze()
            except RuntimeError:
                logger.error("Wasn't able to embed one sequence (probably run out of RAM).")

            # remove special tokens added to end
            embedding = embedding[:-2]

        if not sequence_length == embedding.shape[0]:
            raise SequenceEmbeddingLengthMismatchException()

        return embedding.cpu().detach().numpy().squeeze()

    def embed_many(
        self, sequences: Iterable[str], batch_size: Optional[int] = None
    ) -> Generator[ndarray, None, None]:
        return (self.embed(sequence) for sequence in sequences)

    @staticmethod
    def reduce_per_protein(embedding):
        return embedding.mean(axis=0)

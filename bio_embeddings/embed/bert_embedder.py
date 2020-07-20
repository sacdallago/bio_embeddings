import logging
import re
import tempfile
from pathlib import Path
from typing import Generator, List

import torch
from numpy import ndarray
from transformers import BertModel, BertTokenizer

from bio_embeddings.embed.embedder_interface import EmbedderInterface
from bio_embeddings.embed.helper import embed_batch_berts
from bio_embeddings.utilities import (
    SequenceEmbeddingLengthMismatchException, get_model_directories_from_zip,
)

logger = logging.getLogger(__name__)


class BertEmbedder(EmbedderInterface):
    name = "bert"
    embedding_dimension = 1024
    number_of_layers = 1

    def __init__(self, **kwargs):
        """
        Initialize Bert embedder.

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
        self._model = BertModel.from_pretrained(self._model_directory)
        self._model = self._model.eval()
        self._model = self._model.to(self._device)
        self._tokenizer = BertTokenizer(str(Path(self._model_directory) / 'vocab.txt'), do_lower_case=False)

    @classmethod
    def with_download(cls, **kwargs) -> "BertEmbedder":
        necessary_directories = ['model_directory']

        keep_tempfiles_alive = []
        for directory in necessary_directories:
            if not kwargs.get(directory):
                f = tempfile.mkdtemp()
                keep_tempfiles_alive.append(f)

                get_model_directories_from_zip(path=f, model=cls.name, directory=directory)

                kwargs[directory] = f
        return cls(**kwargs)

    def embed(self, sequence: str) -> ndarray:
        sequence_length = len(sequence)
        sequence = re.sub(r"[UZOB]", "X", sequence)

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

            # remove special tokens added to start/end
            embedding = embedding[1: sequence_length + 1]

        if not sequence_length == embedding.shape[0]:
            raise SequenceEmbeddingLengthMismatchException()

        return embedding.cpu().detach().numpy().squeeze()

    def embed_batch(self, batch: List[str]) -> Generator[ndarray, None, None]:
        return embed_batch_berts(self, batch)

    @staticmethod
    def reduce_per_protein(embedding):
        return embedding.mean(axis=0)

import logging
import re
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Generator, List

import torch
from numpy import ndarray
from transformers import XLNetModel, XLNetTokenizer, XLNetConfig

from bio_embeddings.embed.embedder_interface import Embedder
from bio_embeddings.utilities import get_model_directories_from_zip

logger = logging.getLogger(__name__)


class XLNetEmbedder(Embedder):
    name = "xlnet"
    embedding_dimension = 1024
    number_of_layers = 1

    def __init__(self, **kwargs):
        """
        Initialize XLNet embedder.

        :param model_directory:
        :param use_cpu: overwrite autodiscovery and force CPU use
        """
        super().__init__(**kwargs)

        # Get file locations from kwargs
        self._model_directory = self._options.get("model_directory")

        # make model
        config = XLNetConfig.from_json_file(
            str(Path(self._model_directory) / "config.json")
        )
        # MH: for some reason this has to be set manually. Otherwise, AssertionError during model loading
        config.vocab_size = 37

        self._model = XLNetModel.from_pretrained(
            str(Path(self._model_directory) / "model.ckpt-847000"),
            from_tf=True,
            config=config,
        )
        self._model = self._model.eval()
        self._model = self._model.to(self.device)
        self._tokenizer = XLNetTokenizer(
            str(Path(self._model_directory) / "spm_model.model"), do_lower_case=False
        )

    @classmethod
    def with_download(cls, **kwargs):
        necessary_directories = ["model_directory"]

        for directory in necessary_directories:
            if not kwargs.get(directory):
                f = tempfile.mkdtemp()

                get_model_directories_from_zip(
                    path=f, model=cls.name, directory=directory
                )

                kwargs[directory] = f
        return cls(**kwargs)

    def embed(self, sequence: str) -> ndarray:
        sequence_length = len(sequence)
        sequence = re.sub(r"[UZOBX]", "<unk>", sequence)

        # Tokenize sequence with spaces
        sequence = " ".join(list(sequence))

        # tokenize sequence
        tokenized_sequence = torch.tensor(
            [self._tokenizer.encode(sequence, add_special_tokens=True)]
        ).to(self.device)

        with torch.no_grad():
            # drop batch dimension
            embedding = self._model(tokenized_sequence)[0].squeeze()
            # remove special tokens added to end
            embedding = embedding[:-2]

        assert (
            sequence_length == embedding.shape[0]
        ), f"Sequence length mismatch: {sequence_length} vs {embedding.shape[0]}"

        return embedding.cpu().detach().numpy().squeeze()

    def embed_batch(self, batch: List[str]) -> Generator[ndarray, None, None]:
        # TODO: Actual batching for xlnet
        return (self.embed(sequence) for sequence in batch)

    @staticmethod
    def reduce_per_protein(embedding):
        return embedding.mean(axis=0)

import logging
import re
import tempfile
from itertools import zip_longest
from typing import List, Generator, TypeVar, Union

import torch
from numpy import ndarray
from transformers import BertTokenizer, AlbertTokenizer, BertModel, AlbertModel

from bio_embeddings.embed.embedder_interface import EmbedderInterface
from bio_embeddings.utilities import get_model_directories_from_zip

# https://stackoverflow.com/a/39205612/3549270
T = TypeVar("T", bound="BertBaseEmbedder")

logger = logging.getLogger(__name__)


class BertBaseEmbedder(EmbedderInterface):
    """ Shared code between Bert and ALbert """

    _tokenizer: Union[AlbertTokenizer, BertTokenizer]
    _model: Union[AlbertModel, BertModel]

    @classmethod
    def with_download(cls, **kwargs) -> T:
        necessary_directories = ["model_directory"]

        keep_tempfiles_alive = []
        for directory in necessary_directories:
            if not kwargs.get(directory):
                f = tempfile.mkdtemp()
                keep_tempfiles_alive.append(f)

                get_model_directories_from_zip(
                    path=f, model=cls.name, directory=directory
                )

                kwargs[directory] = f
        return cls(**kwargs)

    def embed_batch(self, batch: List[str]) -> Generator[ndarray, None, None]:
        """ Embed batch code shared between Bert and Albert """
        seq_lens = [len(seq) for seq in batch]
        # Remove rare amino acids
        batch = [re.sub(r"[UZOB]", "X", sequence) for sequence in batch]
        # transformers needs spaces between the amino acids
        batch = [" ".join(list(seq)) for seq in batch]

        ids = self._tokenizer.batch_encode_plus(
            batch, add_special_tokens=True, pad_to_max_length=True
        )

        input_ids = torch.tensor(ids["input_ids"]).to(self.device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(self.device)

        with torch.no_grad():
            embeddings = self._model(input_ids=input_ids, attention_mask=attention_mask)

        embeddings = embeddings[0].cpu().numpy()

        for seq_num, seq_len in zip_longest(range(len(embeddings)), seq_lens):
            # slice off first and last positions (special tokens)
            embedding = embeddings[seq_num][1:-1]
            assert seq_len == embedding.shape[0]
            yield embedding

    @staticmethod
    def reduce_per_protein(embedding):
        return embedding.mean(axis=0)

    def embed(self, sequence: str) -> ndarray:
        sequence_length = len(sequence)
        sequence = re.sub(r"[UZOB]", "X", sequence)

        # Tokenize sequence with spaces
        sequence = " ".join(list(sequence))

        # tokenize sequence
        tokenized_sequence = torch.tensor(
            [self._tokenizer.encode(sequence, add_special_tokens=True)]
        ).to(self.device)

        with torch.no_grad():
            # drop batch dimension
            embedding = self._model(tokenized_sequence)[0].squeeze()
            # remove special tokens added to start/end
            embedding = embedding[1 : sequence_length + 1]

        assert (
            sequence_length == embedding.shape[0]
        ), f"Sequence length mismatch: {sequence_length} vs {embedding.shape[0]}"

        return embedding.cpu().detach().numpy().squeeze()

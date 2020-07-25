import logging
import re
import tempfile
from itertools import zip_longest
from typing import List, Generator, TypeVar, Union, Type

import torch
from numpy import ndarray
from transformers import BertTokenizer, AlbertTokenizer, BertModel, AlbertModel

from bio_embeddings.embed.embedder_interface import EmbedderWithFallback
from bio_embeddings.utilities import get_model_directories_from_zip

# https://stackoverflow.com/a/39205612/3549270
RealBertEmbedder = TypeVar("RealBertEmbedder", bound="BertBaseEmbedder")

logger = logging.getLogger(__name__)


class BertBaseEmbedder(EmbedderWithFallback):
    """ Shared code between Bert and ALbert """

    _tokenizer: Union[AlbertTokenizer, BertTokenizer]
    model: Union[AlbertModel, BertModel]

    @classmethod
    def with_download(cls: Type[RealBertEmbedder], **kwargs) -> RealBertEmbedder:
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

    def _get_fallback_model(self) -> Union[BertModel, AlbertModel]:
        raise NotImplementedError

    def _embed_batch_impl(
        self, batch: List[str], model: Union[BertModel, AlbertModel]
    ) -> Generator[ndarray, None, None]:
        """ Embed batch code shared between Bert and Albert """
        seq_lens = [len(seq) for seq in batch]
        # Remove rare amino acids
        batch = [re.sub(r"[UZOB]", "X", sequence) for sequence in batch]
        # transformers needs spaces between the amino acids
        batch = [" ".join(list(seq)) for seq in batch]

        ids = self._tokenizer.batch_encode_plus(
            batch, add_special_tokens=True, pad_to_max_length=True
        )

        tokenized_sequences = torch.tensor(ids["input_ids"]).to(model.device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(model.device)

        with torch.no_grad():
            embeddings = model(input_ids=tokenized_sequences, attention_mask=attention_mask)

        embeddings = embeddings[0].cpu().numpy()

        for seq_num, seq_len in zip_longest(range(len(embeddings)), seq_lens):
            # slice off first and last positions (special tokens)
            embedding = embeddings[seq_num][1 : seq_len + 1]
            assert (
                seq_len == embedding.shape[0]
            ), f"Sequence length mismatch: {seq_len} vs {embedding.shape[0]}"

            yield embedding

    @staticmethod
    def reduce_per_protein(embedding):
        return embedding.mean(axis=0)

    def embed(self, sequence: str) -> ndarray:
        [embedding] = self.embed_batch([sequence])
        return embedding

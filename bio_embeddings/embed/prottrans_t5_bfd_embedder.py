import logging
import re
from itertools import zip_longest
from pathlib import Path
from typing import List, Generator

import torch
from numpy import ndarray
from transformers import T5Tokenizer, T5Model

from bio_embeddings.embed.embedder_interfaces import EmbedderWithFallback

logger = logging.getLogger(__name__)


class ProtTransT5BFDEmbedder(EmbedderWithFallback):
    """Encoder of the ProtTrans T5 BFD model

    Note that this model alone takes 13GB, so you need a GPU with a lot of memory.
    """

    _model: T5Model
    _decoder: bool = False
    name = "prottrans_t5_bfd"
    embedding_dimension = 1024
    number_of_layers = 1
    _necessary_directories = ["model_directory"]

    def __init__(self, **kwargs):
        """
        Initialize T5 embedder.

        :param model_directory:
        """
        super().__init__(**kwargs)

        self._model_directory = self._options["model_directory"]
        # Until we know whether we need the decoder, let's keep it here as an undocumented option.
        # Should the need arise we can just split this class in to an encoder and a decoder subclass
        # by setting one subclass to _decoder=True and the other to _decoder=False
        self._decoder = self._options.get("decoder", False)

        # make model
        self._model = T5Model.from_pretrained(self._model_directory)
        self._model = self._model.eval().to(self._device)
        self._model_fallback = None
        self._tokenizer = T5Tokenizer(
            str(Path(self._model_directory).joinpath("spiece.model")),
            do_lower_case=False,
        )

    def _get_fallback_model(self) -> T5Model:
        """ Returns the CPU model """
        if not self._model_fallback:
            self._model_fallback = T5Model.from_pretrained(self._model_directory).eval()
        return self._model_fallback

    def _embed_batch_impl(
        self, batch: List[str], model: T5Model
    ) -> Generator[ndarray, None, None]:
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
            embeddings = model(
                input_ids=tokenized_sequences,
                attention_mask=attention_mask,
                decoder_input_ids=tokenized_sequences,
            )

        # See comment in __init__
        if self._decoder:
            embeddings = embeddings[0].cpu().numpy()
        else:
            embeddings = embeddings[2].cpu().numpy()

        for seq_num, seq_len in zip_longest(range(len(embeddings)), seq_lens):
            # slice off last position (special token)
            embedding = embeddings[seq_num][:seq_len]
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

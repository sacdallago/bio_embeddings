import logging
import re
from itertools import zip_longest
from pathlib import Path
from typing import Optional, Generator, List

import torch
from numpy import ndarray
from transformers import XLNetModel, XLNetTokenizer

from bio_embeddings.embed.embedder_interfaces import EmbedderInterface

logger = logging.getLogger(__name__)


class ProtTransXLNetUniRef100Embedder(EmbedderInterface):
    """ProtTrans-XLNet-UniRef100 Embedder (ProtXLNet)

    Elnaggar, Ahmed, et al. "ProtTrans: Towards Cracking the Language of Life's
    Code Through Self-Supervised Deep Learning and High Performance Computing."
    arXiv preprint arXiv:2007.06225 (2020). https://arxiv.org/abs/2007.06225
    """

    name = "prottrans_xlnet_uniref100"
    embedding_dimension = 1024
    number_of_layers = 1
    _model: XLNetModel
    _model_fallback: Optional[XLNetModel]
    necessary_directories = ["model_directory"]

    def __init__(self, **kwargs):
        """
        Initialize XLNet embedder.

        :param model_directory:
        """
        super().__init__(**kwargs)

        # Get file locations from kwargs
        self.model_directory = self._options["model_directory"]

        # 512 is from https://github.com/agemagician/ProtTrans/blob/master/Embedding/PyTorch/Advanced/ProtXLNet.ipynb
        self._model = (
            XLNetModel.from_pretrained(self.model_directory, mem_len=512)
            .to(self._device)
            .eval()
        )
        self._model_fallback = None

        # sentence piece model
        # A standard text tokenizer which creates the input for NNs trained on text.
        # This one is just indexing single amino acids because we only have words of L=1.
        spm_model = str(Path(self.model_directory).joinpath("spm_model.model"))
        self._tokenizer = XLNetTokenizer.from_pretrained(spm_model, do_lower_case=False)

    def embed(self, sequence: str) -> ndarray:
        [embedding] = self.embed_batch([sequence])
        return embedding

    def embed_batch(self, batch: List[str]) -> Generator[ndarray, None, None]:
        seq_lens = [len(seq) for seq in batch]
        # transformers needs spaces between the amino acids
        batch = [" ".join(list(seq)) for seq in batch]
        # Remove rare amino acids
        batch = [re.sub(r"[UZOBX]", "<unk>", sequence) for sequence in batch]

        ids = self._tokenizer.batch_encode_plus(
            batch, add_special_tokens=True, padding="longest"
        )

        tokenized_sequences = torch.tensor(ids["input_ids"]).to(self._model.device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(self._model.device)

        with torch.no_grad():
            embeddings, memory = self._model(
                input_ids=tokenized_sequences,
                attention_mask=attention_mask,
                mems=None,
                return_dict=False,
            )

        embeddings = embeddings.cpu().numpy()

        for seq_num, seq_len in zip_longest(range(len(embeddings)), seq_lens):
            attention_len = (attention_mask[seq_num] == 1).sum()
            padded_seq_len = len(attention_mask[seq_num])
            embedding = embeddings[seq_num][
                padded_seq_len - attention_len : padded_seq_len - 2
            ]
            assert (
                seq_len == embedding.shape[0]
            ), f"Sequence length mismatch: {seq_len} vs {embedding.shape[0]}"
            yield embedding

    @staticmethod
    def reduce_per_protein(embedding):
        return embedding.mean(axis=0)

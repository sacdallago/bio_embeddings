import logging
from pathlib import Path
from typing import Optional, List, Generator

import torch
from numpy import ndarray
from torch import nn
from torch import tensor
from transformers import BertModel, BertTokenizer

from bio_embeddings.embed.prottrans_base_embedder import ProtTransBertBaseEmbedder
from bio_embeddings.utilities import get_model_file

logger = logging.getLogger(__name__)


class Tucker(nn.Module):
    """Tucker is a contrastive learning model trained to distinguish CATH superfamilies.

    It consumes prottrans_bert_bfd embeddings and reduces the embedding dimensionality from 1024 to 128.
    See https://www.biorxiv.org/content/10.1101/2021.01.21.427551v1
    """

    def __init__(self):
        super(Tucker, self).__init__()
        self.tucker = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
        )

    @staticmethod
    def from_file(model_file: Path, device: torch.device) -> "Tucker":
        model = Tucker()
        model.load_state_dict(torch.load(model_file, map_location=device)["state_dict"])
        model.eval()
        return model.to(device)

    def single_pass(self, x: tensor) -> tensor:
        return self.tucker(x)


class ProtTransBertBFDEmbedder(ProtTransBertBaseEmbedder):
    """ProtTrans-Bert-BFD Embedder (ProtBert-BFD)

    Elnaggar, Ahmed, et al. "ProtTrans: Towards Cracking the Language of Life's
    Code Through Self-Supervised Deep Learning and High Performance Computing."
    arXiv preprint arXiv:2007.06225 (2020). https://arxiv.org/abs/2007.06225
    """

    _model: BertModel
    _tucker: Optional[Tucker] = None
    name = "prottrans_bert_bfd"
    embedding_dimension = 1024
    number_of_layers = 1

    def __init__(self, **kwargs):
        """
        Initialize Bert embedder.

        :param model_directory:
        """
        super().__init__(**kwargs)

        self._model_directory = self._options["model_directory"]

        if self._options.get("use_tucker"):
            if "pb_tucker_model_file" not in self._options:
                model_file = get_model_file("pb_tucker", "model_file")
            else:
                model_file = self._options["model_file"]
            self._tucker = Tucker.from_file(model_file, self._device)

        # make model
        self._model = BertModel.from_pretrained(self._model_directory)
        self._model = self._model.eval().to(self._device)
        self._model_fallback = None
        self._tokenizer = BertTokenizer(
            str(Path(self._model_directory) / "vocab.txt"), do_lower_case=False
        )

    def _embed_batch_impl(
        self, batch: List[str], model: BertModel
    ) -> Generator[ndarray, None, None]:
        """Handle tucker"""
        for embedding in super()._embed_batch_impl(batch, model):
            if self._tucker:
                with torch.no_grad():
                    # The back-and-forth is a bit awkward, but it shouldn't be a bottleneck
                    yield self._tucker.single_pass(
                        torch.tensor(embedding, device=self._device)
                    ).cpu().numpy()
            else:
                yield embedding

    def _get_fallback_model(self) -> BertModel:
        """ Returns the CPU model """
        if not self._model_fallback:
            self._model_fallback = BertModel.from_pretrained(
                self._model_directory
            ).eval()
        return self._model_fallback

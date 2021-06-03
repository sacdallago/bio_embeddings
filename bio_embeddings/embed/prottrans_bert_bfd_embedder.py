import logging
from pathlib import Path

from transformers import BertModel, BertTokenizer

from bio_embeddings.embed.prottrans_base_embedder import ProtTransBertBaseEmbedder

logger = logging.getLogger(__name__)


class ProtTransBertBFDEmbedder(ProtTransBertBaseEmbedder):
    """ProtTrans-Bert-BFD Embedder (ProtBert-BFD)

    Elnaggar, Ahmed, et al. "ProtTrans: Towards Cracking the Language of Life's
    Code Through Self-Supervised Deep Learning and High Performance Computing."
    arXiv preprint arXiv:2007.06225 (2020). https://arxiv.org/abs/2007.06225
    """

    _model: BertModel
    name = "prottrans_bert_bfd"
    embedding_dimension = 1024
    number_of_layers = 1

    def __init__(self, **kwargs):
        """Initialize Bert embedder.

        :param model_directory:
        :param half_precision_model:
        """
        super().__init__(**kwargs)

        self._model_directory = self._options["model_directory"]
        self._half_precision_model = self._options.get("half_precision_model", False)

        # make model
        self._model = BertModel.from_pretrained(self._model_directory)
        # Compute in half precision, which is a lot faster and saves us half the memory
        if self._half_precision_model:
            self._model = self._model.half()
        self._model = self._model.eval().to(self._device)
        self._model_fallback = None
        self._tokenizer = BertTokenizer(
            str(Path(self._model_directory) / "vocab.txt"), do_lower_case=False
        )

    def _get_fallback_model(self) -> BertModel:
        """ Returns the CPU model """
        if not self._model_fallback:
            self._model_fallback = BertModel.from_pretrained(
                self._model_directory
            ).eval()
        return self._model_fallback

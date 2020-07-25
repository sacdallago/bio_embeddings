import logging
from pathlib import Path

from transformers import BertModel, BertTokenizer

from bio_embeddings.embed.bert_base_embedder import BertBaseEmbedder

logger = logging.getLogger(__name__)


class BertEmbedder(BertBaseEmbedder):
    name = "bert"
    embedding_dimension = 1024
    number_of_layers = 1

    def __init__(self, **kwargs):
        """
        Initialize Bert embedder.

        :param model_directory:
        :param use_cpu: overwrite autodiscovery and force CPU use
        """
        super().__init__(**kwargs)

        # Get file locations from kwargs
        self._model_directory = self._options.get("model_directory")

        # make model
        self._model = BertModel.from_pretrained(self._model_directory)
        self._model = self._model.eval().to(self.device)
        self._tokenizer = BertTokenizer(
            str(Path(self._model_directory) / "vocab.txt"), do_lower_case=False
        )

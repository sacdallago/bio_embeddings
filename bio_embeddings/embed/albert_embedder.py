from pathlib import Path

from transformers import AlbertModel, AlbertTokenizer

from bio_embeddings.embed.bert_base_embedder import BertBaseEmbedder


class AlbertEmbedder(BertBaseEmbedder):
    name = "albert"
    embedding_dimension = 4096
    number_of_layers = 1

    def __init__(self, **kwargs):
        """
        Initialize Albert embedder.

        :param model_directory:
        :param use_cpu: overwrite autodiscovery and force CPU use
        """
        super().__init__(**kwargs)

        # Get file locations from kwargs
        self._model_directory = self._options.get("model_directory")

        # make model
        self.model = AlbertModel.from_pretrained(self._model_directory)
        self.model = self.model.eval().to(self._device)
        self.model_fallback = None
        self._tokenizer = AlbertTokenizer(
            str(Path(self._model_directory) / "albert_vocab_model.model"),
            do_lower_case=False,
        )

    def _get_fallback_model(self) -> AlbertModel:
        """ Returns the CPU model """
        if not self.model_fallback:
            self.model_fallback = AlbertModel.from_pretrained(
                self._model_directory
            ).eval()
        return self.model_fallback

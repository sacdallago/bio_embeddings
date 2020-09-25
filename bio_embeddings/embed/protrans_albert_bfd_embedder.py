from pathlib import Path

from transformers import AlbertModel, AlbertTokenizer

from bio_embeddings.embed.prottrans_bert_base_embedder import BertBaseEmbedder


class ProtTransAlbertBFDEmbedder(BertBaseEmbedder):
    _model: AlbertModel
    name = "prottrans_albert_bfd"
    embedding_dimension = 4096
    number_of_layers = 1

    def __init__(self, **kwargs):
        """
        Initialize Albert embedder.

        :param model_directory:
        """
        super().__init__(**kwargs)

        # Get file locations from kwargs
        self._model_directory = self._options["model_directory"]

        # make model
        self._model = AlbertModel.from_pretrained(self._model_directory)
        self._model = self._model.eval().to(self._device)
        self._model_fallback = None
        self._tokenizer = AlbertTokenizer(
            str(Path(self._model_directory) / "albert_vocab_model.model"),
            do_lower_case=False,
        )

    def _get_fallback_model(self) -> AlbertModel:
        """ Returns the CPU model """
        if not self._model_fallback:
            self._model_fallback = AlbertModel.from_pretrained(
                self._model_directory
            ).eval()
        return self._model_fallback

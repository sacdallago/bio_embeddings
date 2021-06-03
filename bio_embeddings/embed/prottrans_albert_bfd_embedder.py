from pathlib import Path

from transformers import AlbertModel, AlbertTokenizer

from bio_embeddings.embed.prottrans_base_embedder import ProtTransBertBaseEmbedder


class ProtTransAlbertBFDEmbedder(ProtTransBertBaseEmbedder):
    """ProtTrans-Albert-BFD Embedder (ProtAlbert-BFD)

    Elnaggar, Ahmed, et al. "ProtTrans: Towards Cracking the Language of Life's
    Code Through Self-Supervised Deep Learning and High Performance Computing."
    arXiv preprint arXiv:2007.06225 (2020). https://arxiv.org/abs/2007.06225
    """

    _model: AlbertModel
    name = "prottrans_albert_bfd"
    embedding_dimension = 4096
    number_of_layers = 1

    def __init__(self, **kwargs):
        """Initialize Albert embedder.

        :param model_directory:
        :param half_precision_model:
        """
        super().__init__(**kwargs)

        self._model_directory = self._options["model_directory"]
        self._half_precision_model = self._options.get("half_precision_model", False)

        # make model
        self._model = AlbertModel.from_pretrained(self._model_directory)
        # Compute in half precision, which is a lot faster and saves us half the memory
        if self._half_precision_model:
            self._model = self._model.half()
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

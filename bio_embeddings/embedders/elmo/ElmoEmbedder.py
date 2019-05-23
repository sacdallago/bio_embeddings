import torch
from bio_embeddings.embedders.EmbedderInterface import EmbedderInterface, NoEmbeddingException
from bio_embeddings.embedders.utlitieis import
from bio_embeddings.utilities import Logger, get_defaults
from allennlp.commands.elmo import ElmoEmbedder as _ElmoEmbedder


class ElmoEmbedder(EmbedderInterface):

    def __init__(self, weights_file, options_file):
        super().__init__(weights_file, options_file)

        # use GPU if available, otherwise run on CPU
        if torch.cuda.is_available():
            Logger.log("CUDA available")
            _cuda_device = 0
        else:
            Logger.log("CUDA NOT available")
            _cuda_device = -1

        if self._weights_file is None or self._options_file is None:
            self._weight_file, self._options_file = get_defaults('elmov1')

        self._model = _ElmoEmbedder(weight_file=self._weight_file, options_file=self._options_file, cuda_device=_cuda_device)
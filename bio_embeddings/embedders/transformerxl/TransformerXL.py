from bio_embeddings.embedders.EmbedderInterface import EmbedderInterface
from bio_embeddings.embedders.transformerxl.mem_transformer import MemTransformerLM
import os

class TransformerXLEmbedder(EmbedderInterface):

    def __init__(self, **kwargs):
        """

        :param model: string, options: "base", "large". Default: "base"
        """
        super().__init__()

        # see issue #152
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        #
        # self._model = self.get_transformerxl_models(model_name)
        # self._vocabulary = self.get_vocab(model_name)

        pass


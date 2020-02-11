import torch
import os
import sys
from bio_embeddings.embed.EmbedderInterface import EmbedderInterface

# TODO: this is neccessary to import mem_transformer --> prettify when packaged!
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
sys.path.append(os.path.join(current_path, 'utils'))


class TransformerXLEmbedder(EmbedderInterface):
    def __init__(self, **kwargs):
        """

        :param model: string, options: "base", "large". Default: "base"
        :param model_file: path to model file. If missing, will download base model + vocabulary
        :param vocabulary_file: path to vocabulary file. If missing, will download base model + vocabulary
        """
        super().__init__()

        # # see issue #152
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = "7"

        self._options = kwargs

        self._model_size = self._options.get('model', 'base')

        if self._model_size not in ['base', 'large']:
            raise InvalidModelSizeException

        self._model_file = self._options.get('model_file')
        self._vocabulary_file = self._options.get('vocabulary_file')

        # if self._model_file is None or self._vocabulary_file is None:
        #     self._temp_model_file, self._temp_vocabulary_file = get_model_parameters('transformer_{}'.format(self._model_size))
        #     self._model_file, self._vocabulary_file = self._temp_model_file.name, self._temp_vocabulary_file.name

        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self._model = torch.load(self._model_file, map_location=self._device)
        self._model = self._model.to(self._device)
        self._model = self._model.eval()

        self._vocabulary = torch.load(self._vocabulary_file, map_location=self._device)

        pass

    def get_features(self, embedding=None):
        raise NotImplementedError

    def embed(self, sequence):
        self._sequence = sequence

        encoded_data = self._vocabulary.encode_sents(self._sequence)
        encoded_data = torch.LongTensor(encoded_data).to(self._device)
        encoded_data = encoded_data.unsqueeze(1)

        with torch.no_grad():
            # Predict hidden states features for last layer
            self._embedding, _ = self._model(encoded_data)

        self._embedding = self._embedding.cpu().detach().numpy().squeeze()
        return self._embedding


class InvalidModelSizeException(Exception):
    """
    Passed an invalid model size
    """

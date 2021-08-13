import logging
import numpy

import torch
import collections

from typing import List, Union
from numpy import ndarray
from enum import Enum

from bio_embeddings.extract.annotations import Conservation
from bio_embeddings.extract.prott5cons.conservation_cnn import ConservationCNN
from bio_embeddings.utilities import get_device, get_model_file

logger = logging.getLogger(__name__)

# Label mappings
_conservation_labels = {
    0: Conservation.cons_1,
    1: Conservation.cons_2,
    2: Conservation.cons_3,
    3: Conservation.cons_4,
    4: Conservation.cons_5,
    5: Conservation.cons_6,
    6: Conservation.cons_7,
    7: Conservation.cons_8,
    8: Conservation.cons_9,
}


BasicConservationResult = collections.namedtuple('BasicConservationResult', 'conservation')

class ProtT5consAnnotationExtractor():
    necessary_files = ["model_file"]

    def __init__(self, model_type: str, device: Union[None, str, torch.device] = None, **kwargs):
        """
        Initialize annotation extractor. Must define non-positional arguments for paths of files.

        :param model_file: path of conservation inference model checkpoint file (only CNN-architecture from paper)
        """

        self._options = kwargs
        self._model_type = model_type
        self._device = get_device(device)

        # Create un-trained (raw) model and ensure self._model_type is valid
        self._conservation_model = ConservationCNN().to(self._device)

        # Download the checkpoint files if needed
        if not self._options.get('model_file'):
            self._options['model_file'] = get_model_file(model=f"prott5cons", file="model_file")

        self.model_file = self._options['model_file']

        # load pre-trained weights for annotation machines
        conservation_state = torch.load(self.model_file, map_location=self._device)

        # load pre-trained weights into raw model
        self._conservation_model.load_state_dict(conservation_state['state_dict'])

        # ensure that model is in evaluation mode (important for batchnorm and dropout)
        self._conservation_model.eval()

    # This will be useful if we want to stitch together conservation prediction with SAV effect prediction (we'll need raw output)
    def get_raw_predictions(self, raw_embedding: ndarray) -> ndarray:
        raw_embedding = raw_embedding.astype(numpy.float32)  # For T5 fp16
        embedding = torch.tensor(raw_embedding).to(self._device)
        # Pass embeddings to model to produce predictions
        yhat_conservation = self._conservation_model(embedding)
        return yhat_conservation

    def get_conservation(self, raw_embedding: ndarray) -> BasicConservationResult:
        raw_embedding = raw_embedding.astype(numpy.float32)  # For T5 fp16
        embedding = torch.tensor(raw_embedding).to(self._device)
        # Pass embeddings to model to produce predictions
        yhat_conservation = self._conservation_model(embedding)
        # Map raw class predictions (integers) to class labels (strings)
        pred_conservation = self._class2label(_conservation_labels, yhat_conservation)

        return BasicConservationResult(conservation=pred_conservation)


    @staticmethod
    def _class2label(label_dict, yhat) -> List[Enum]:
        # get index of output node with max. activation (=predicted class)
        class_indices = torch.max(yhat, dim=1)[1].squeeze()
        return [label_dict[class_idx.item()] for class_idx in class_indices]

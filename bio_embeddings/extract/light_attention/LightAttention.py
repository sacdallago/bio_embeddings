import logging
import torch

from typing import List, Union
from numpy import ndarray
from enum import Enum

from bio_embeddings.extract.annotations import Location, Membrane
from bio_embeddings.extract.basic import SubcellularLocalizationAndMembraneBoundness
from bio_embeddings.extract.light_attention.light_attention_model import LightAttention
from bio_embeddings.utilities import get_device, get_model_file

logger = logging.getLogger(__name__)

# Label mappings
_loc_labels = {
    0: Location.CELL_MEMBRANE,
    1: Location.CYTOPLASM,
    2: Location.ENDOPLASMATIC_RETICULUM,
    3: Location.GOLGI_APPARATUS,
    4: Location.LYSOSOME_OR_VACUOLE,
    5: Location.MITOCHONDRION,
    6: Location.NUCLEUS,
    7: Location.PEROXISOME,
    8: Location.PLASTID,
    9: Location.EXTRACELLULAR
}

# the light attention models are trained with 0 as index for membrane. For the basic models this is the other way around
_mem_labels = {
    0: Membrane.MEMBRANE,
    1: Membrane.SOLUBLE
}


class LightAttentionAnnotationExtractor(object):
    necessary_files = ["subcellular_location_checkpoint_file", 'membrane_checkpoint_file']

    def __init__(self, device: Union[None, str, torch.device] = None, **kwargs):
        """
        Initialize annotation extractor. Must define non-positional arguments for paths of files.

        :param membrane_checkpoint_file: path of secondary structure inference model checkpoint file
        :param subcellular_location_checkpoint_file: path of the subcellular location inference model checkpoint file
        """

        self._options = kwargs
        self._device = get_device(device)

        # Create un-trained (raw) model
        self._subcellular_location_model = LightAttention(output_dim=10).to(self._device)
        self._membrane_model = LightAttention(output_dim=2).to(self._device)

        # Download the checkpoint files if needed
        for file in self.necessary_files:
            if not self._options.get(file):
                self._options[file] = get_model_file(model="light_attention_annotations_extractors", file=file)

        self._subcellular_location_checkpoint_file = self._options.get('subcellular_location_checkpoint_file')
        self._membrane_checkpoint_file = self._options.get('membrane_checkpoint_file')
        self._device = get_device(device)

        # load pre-trained weights for annotation machines
        subcellular_state = torch.load(self._subcellular_location_checkpoint_file, map_location=self._device)
        membrane_state = torch.load(self._membrane_checkpoint_file, map_location=self._device)

        # load pre-trained weights into raw model
        self._subcellular_location_model.load_state_dict(subcellular_state['state_dict'])
        self._membrane_model.load_state_dict(membrane_state['state_dict'])

        # ensure that model is in evaluation mode (important for batchnorm and dropout)
        self._subcellular_location_model.eval()
        self._membrane_model.eval()

    def get_subcellular_location(self, raw_embedding: ndarray) -> SubcellularLocalizationAndMembraneBoundness:
        '''
        :param raw_embedding: np array of [sequence_length, 1024]

        :eturns: SubcellularLocalizationAndMembraneBoundness with predictions for localization and membrane bound or not
        '''
        # turn to tensor and add singleton batch dimension
        embedding = torch.tensor(raw_embedding).to(self._device)[None, ...]

        yhat_loc = self._subcellular_location_model(embedding)
        yhat_mem = self._membrane_model(embedding)

        pred_loc = _loc_labels[torch.max(yhat_loc, dim=1)[1].item()]  # get index of output node with max. activation,
        pred_mem = _mem_labels[torch.max(yhat_mem, dim=1)[1].item()]  # this corresponds to the predicted class

        return SubcellularLocalizationAndMembraneBoundness(localization=pred_loc, membrane=pred_mem)

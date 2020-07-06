import logging
import torch

from typing import List
from numpy import ndarray

from bio_embeddings.extract.features import Location, Membrane, Disorder, SecondaryStructure, FeatureInterface
from bio_embeddings.extract.seqvec.feature_inference_models import SUBCELL_FNN, SECSTRUCT_CNN

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

_mem_labels = {
    0: Membrane.SOLUBLE,
    1: Membrane.MEMBRANE
}

_dssp8_labels = {
    0: SecondaryStructure.THREE_HELIX,
    1: SecondaryStructure.ALPHA_HELIX,
    2: SecondaryStructure.FIVE_HELIX,
    3: SecondaryStructure.ISOLATED_BETA_BRIDGE,
    4: SecondaryStructure.EXTENDED_STRAND,
    5: SecondaryStructure.BEND,
    6: SecondaryStructure.TURN,
    7: SecondaryStructure.IRREGULAR
}

_dssp3_labels = {
    0: SecondaryStructure.ALPHA_HELIX,
    1: SecondaryStructure.EXTENDED_STRAND,
    2: SecondaryStructure.IRREGULAR
}

_disor_labels = {
    0: Disorder.ORDER,
    1: Disorder.DISORDER
}


class SeqVecFeatureExtractor(object):

    def __init__(self, **kwargs):
        """
        Initialize SeqVec feature extractor. Must define non-positional arguments for paths of files.

        :param secondary_structure_checkpoint_file: path of secondary structure checkpoint file
        :param subcellular_location_checkpoint_file: path of the subcellular location checkpoint file
        """

        self._options = kwargs

        self._secondary_structure_checkpoint_file = self._options.get('secondary_structure_checkpoint_file')
        self._subcellular_location_checkpoint_file = self._options.get('subcellular_location_checkpoint_file')

        # use GPU if available, otherwise run on CPU
        # !important: GPU visibility can easily be hidden using this env variable: CUDA_VISIBLE_DEVICES=""
        # This is especially useful if using an old CUDA device which is not supported by pytorch!

        # TODO: this needs to be done better!!
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Read in pre-trained model

        # Create un-trained (raw) model
        self._subcellular_location_model = SUBCELL_FNN().to(self._device)
        self._secondary_structure_model = SECSTRUCT_CNN().to(self._device)

        if torch.cuda.is_available():
            logger.info("CUDA available")

            # load pre-trained weights for feature machines
            subcellular_state = torch.load(self._subcellular_location_checkpoint_file)
            secondary_structure_state = torch.load(self._secondary_structure_checkpoint_file)
        else:
            logger.info("CUDA NOT available")

            # load pre-trained weights for feature machines
            subcellular_state = torch.load(self._subcellular_location_checkpoint_file, map_location='cpu')
            secondary_structure_state = torch.load(self._secondary_structure_checkpoint_file, map_location='cpu')
            pass

        # load pre-trained weights into raw model
        self._subcellular_location_model.load_state_dict(subcellular_state['state_dict'])
        self._secondary_structure_model.load_state_dict(secondary_structure_state['state_dict'])

        # ensure that model is in evaluation mode (important for batchnorm and dropout)
        self._subcellular_location_model.eval()
        self._secondary_structure_model.eval()

    def get_subcellular_location(self, raw_embedding: ndarray):
        embedding = torch.tensor(raw_embedding).to(self._device).sum(dim=0).mean(dim=0, keepdim=True)
        yhat_loc, yhat_mem = self._subcellular_location_model(embedding)

        pred_loc = _loc_labels[torch.max(yhat_loc, dim=1)[1].item()]  # get index of output node with max. activation,
        pred_mem = _mem_labels[torch.max(yhat_mem, dim=1)[1].item()]  # this corresponds to the predicted class

        return pred_loc, pred_mem

    def get_secondary_structure(self, embedding):

        embedding = torch.tensor(embedding).to(self._device).sum(dim=0, keepdim=True).permute(0, 2, 1).unsqueeze(dim=-1)
        yhat_dssp3, yhat_dssp8, yhat_disor = self._secondary_structure_model(embedding)

        pred_dssp3 = self._class2label(_dssp3_labels, yhat_dssp3)
        pred_dssp8 = self._class2label(_dssp8_labels, yhat_dssp8)
        pred_disor = self._class2label(_disor_labels, yhat_disor)

        return pred_dssp3, pred_dssp8, pred_disor

    @staticmethod
    def _class2label(label_dict, yhat) -> List[FeatureInterface]:
        # get index of output node with max. activation (=predicted class)
        class_indices = torch.max(yhat, dim=1)[1].squeeze()
        return [label_dict[class_idx.item()] for class_idx in class_indices]

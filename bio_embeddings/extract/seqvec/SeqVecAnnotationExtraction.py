import logging
import torch
import collections

from typing import List
from numpy import ndarray
from enum import Enum

from bio_embeddings.extract.annotations import Location, Membrane, Disorder, SecondaryStructure
from bio_embeddings.extract.seqvec.annotation_inference_models import SUBCELL_FNN, SECSTRUCT_CNN

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

SecondaryStructureResult = collections.namedtuple('SecondaryStructure', 'DSSP3 DSSP8 disorder')
SubcellularLocalizationResult = collections.namedtuple('SubcellularLocalization', 'localization membrane')
SeqVecExtractedAnnotations = collections.namedtuple('SeqVecExtractedAnnotations', 'DSSP3 DSSP8 disorder localization membrane')


class SeqVecAnnotationExtractor(object):

    def __init__(self, **kwargs):
        """
        Initialize SeqVec annotation extractor. Must define non-positional arguments for paths of files.

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

            # load pre-trained weights for annotation machines
            subcellular_state = torch.load(self._subcellular_location_checkpoint_file)
            secondary_structure_state = torch.load(self._secondary_structure_checkpoint_file)
        else:
            logger.info("CUDA NOT available")

            # load pre-trained weights for annotation machines
            subcellular_state = torch.load(self._subcellular_location_checkpoint_file, map_location='cpu')
            secondary_structure_state = torch.load(self._secondary_structure_checkpoint_file, map_location='cpu')
            pass

        # load pre-trained weights into raw model
        self._subcellular_location_model.load_state_dict(subcellular_state['state_dict'])
        self._secondary_structure_model.load_state_dict(secondary_structure_state['state_dict'])

        # ensure that model is in evaluation mode (important for batchnorm and dropout)
        self._subcellular_location_model.eval()
        self._secondary_structure_model.eval()

    def get_subcellular_location(self, raw_embedding: ndarray) -> SubcellularLocalizationResult:
        embedding = torch.tensor(raw_embedding).to(self._device).sum(dim=0).mean(dim=0, keepdim=True)
        yhat_loc, yhat_mem = self._subcellular_location_model(embedding)

        pred_loc = _loc_labels[torch.max(yhat_loc, dim=1)[1].item()]  # get index of output node with max. activation,
        pred_mem = _mem_labels[torch.max(yhat_mem, dim=1)[1].item()]  # this corresponds to the predicted class

        return SubcellularLocalizationResult(localization=pred_loc, membrane=pred_mem)

    def get_secondary_structure(self, raw_embedding: ndarray) -> SecondaryStructureResult:
        embedding = torch.tensor(raw_embedding).to(self._device).sum(dim=0, keepdim=True).permute(0, 2, 1).unsqueeze(dim=-1)
        yhat_dssp3, yhat_dssp8, yhat_disor = self._secondary_structure_model(embedding)

        pred_dssp3 = self._class2label(_dssp3_labels, yhat_dssp3)
        pred_dssp8 = self._class2label(_dssp8_labels, yhat_dssp8)
        pred_disor = self._class2label(_disor_labels, yhat_disor)

        return SecondaryStructureResult(DSSP3=pred_dssp3, DSSP8=pred_dssp8, disorder=pred_disor)

    def get_annotations(self, raw_embedding: ndarray) -> SeqVecExtractedAnnotations:
        secstruct = self.get_secondary_structure(raw_embedding)
        subcell = self.get_subcellular_location(raw_embedding)

        return SeqVecExtractedAnnotations(disorder=secstruct.disorder, DSSP8=secstruct.DSSP8,
                                          DSSP3=secstruct.DSSP3, localization=subcell.localization,
                                          membrane=subcell.membrane)

    @staticmethod
    def _class2label(label_dict, yhat) -> List[Enum]:
        # get index of output node with max. activation (=predicted class)
        class_indices = torch.max(yhat, dim=1)[1].squeeze()
        return [label_dict[class_idx.item()] for class_idx in class_indices]

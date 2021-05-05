import logging
import numpy

import torch
import collections

from typing import List, Union
from numpy import ndarray
from enum import Enum

from bio_embeddings.extract.annotations import Location, Membrane, Disorder, SecondaryStructure
from bio_embeddings.extract.basic.annotation_inference_models import SUBCELL_FNN, SECSTRUCT_CNN
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

BasicSecondaryStructureResult = collections.namedtuple('BasicSecondaryStructureResult', 'DSSP3 DSSP8 disorder')
SubcellularLocalizationAndMembraneBoundness = collections.namedtuple('SubcellularLocalizationAndMembraneBoundness', 'localization membrane')
BasicExtractedAnnotations = collections.namedtuple('BasicExtractedAnnotations', 'DSSP3 DSSP8 disorder localization membrane')


class BasicAnnotationExtractor(object):
    necessary_files = ["secondary_structure_checkpoint_file", "subcellular_location_checkpoint_file"]

    def __init__(self, model_type: str, device: Union[None, str, torch.device] = None, **kwargs):
        """
        Initialize annotation extractor. Must define non-positional arguments for paths of files.

        :param secondary_structure_checkpoint_file: path of secondary structure inference model checkpoint file
        :param subcellular_location_checkpoint_file: path of the subcellular location inference model checkpoint file
        """

        self._options = kwargs
        self._model_type = model_type
        self._device = get_device(device)

        # Create un-trained (raw) model and ensure self._model_type is valid
        if self._model_type == "seqvec_from_publication":
            self._subcellular_location_model = SUBCELL_FNN().to(self._device)
        elif self._model_type == "bert_from_publication" or self._model_type == "t5_xl_u50_from_publication": # Drop batchNorm for ProtTrans models
            self._subcellular_location_model = SUBCELL_FNN(use_batch_norm=False).to(self._device)
        else:
            print("You first need to define your custom model architecture.")
            raise NotImplementedError

        # Download the checkpoint files if needed
        for file in self.necessary_files:
            if not self._options.get(file):
                self._options[file] = get_model_file(model=f"{self._model_type}_annotations_extractors", file=file)

        self._secondary_structure_checkpoint_file = self._options['secondary_structure_checkpoint_file']
        self._subcellular_location_checkpoint_file = self._options['subcellular_location_checkpoint_file']

        # Read in pre-trained model
        self._secondary_structure_model = SECSTRUCT_CNN().to(self._device)

        # load pre-trained weights for annotation machines
        subcellular_state = torch.load(self._subcellular_location_checkpoint_file, map_location=self._device)
        secondary_structure_state = torch.load(self._secondary_structure_checkpoint_file, map_location=self._device)

        # load pre-trained weights into raw model
        self._subcellular_location_model.load_state_dict(subcellular_state['state_dict'])
        self._secondary_structure_model.load_state_dict(secondary_structure_state['state_dict'])

        # ensure that model is in evaluation mode (important for batchnorm and dropout)
        self._subcellular_location_model.eval()
        self._secondary_structure_model.eval()

    def get_subcellular_location(self, raw_embedding: ndarray) -> SubcellularLocalizationAndMembraneBoundness:
        raw_embedding = raw_embedding.astype(numpy.float32)  # For T5 fp16
        # Reduce embedding to fixed size, per-sequence (aka: Lx3x2014 --> 1024).
        # This is similar to embedder.reduce_per_protein(),
        # but more efficient since may be run in GPU (see self._device)

        # TODO: xxmh I forgot that SeqVec requires different pooling to derive fixed size rep.
        #   SeqVec requires summing over 3 layers, ProtTrans models only extract last layers
        #   Quick&Dirty solution is to check for shape of embedding tensors as SeqVec has 3 dims,
        #   while ProtTrans should only have 2 dims.
        #   Better way would be to access some internal variable (probably I just missed this flag)
        #   XXCD: can check embedder type via protol in embed config, but this may become complicated...
        if self._model_type == "seqvec_from_publication":
            # SeqVec case
            embedding = torch.tensor(raw_embedding).to(self._device).sum(dim=0).mean(dim=0, keepdim=True)
        elif self._model_type == "bert_from_publication" or self._model_type == "t5_xl_u50_from_publication":
            # Bert/T5 case
            embedding = torch.tensor(raw_embedding).to(self._device).mean(dim=0, keepdim=True)
        else:
            raise NotImplementedError

        yhat_loc, yhat_mem = self._subcellular_location_model(embedding)

        pred_loc = _loc_labels[torch.max(yhat_loc, dim=1)[1].item()]  # get index of output node with max. activation,
        pred_mem = _mem_labels[torch.max(yhat_mem, dim=1)[1].item()]  # this corresponds to the predicted class

        return SubcellularLocalizationAndMembraneBoundness(localization=pred_loc, membrane=pred_mem)

    def get_secondary_structure(self, raw_embedding: ndarray) -> BasicSecondaryStructureResult:
        raw_embedding = raw_embedding.astype(numpy.float32)  # For T5 fp16
        # same as for subcell loc.: SeqVec requires summing over layers while ProtTrans models only extract last layers
        if self._model_type == "seqvec_from_publication":
            # SeqVec case
            embedding = torch.tensor(raw_embedding).to(self._device).sum(dim=0, keepdim=True).permute(0, 2, 1).unsqueeze(dim=-1)
        elif self._model_type == "bert_from_publication" or self._model_type == "t5_xl_u50_from_publication":
            # Bert/T5 case
            # Flip dimensions for ProtTrans models in order to make feature dimension the first dimension
            embedding = torch.tensor(raw_embedding).to(self._device).T[None, :, :, None]
        else:
            raise NotImplementedError

        yhat_dssp3, yhat_dssp8, yhat_disor = self._secondary_structure_model(embedding)

        pred_dssp3 = self._class2label(_dssp3_labels, yhat_dssp3)
        pred_dssp8 = self._class2label(_dssp8_labels, yhat_dssp8)
        pred_disor = self._class2label(_disor_labels, yhat_disor)

        return BasicSecondaryStructureResult(DSSP3=pred_dssp3, DSSP8=pred_dssp8, disorder=pred_disor)

    def get_annotations(self, raw_embedding: ndarray) -> BasicExtractedAnnotations:
        secstruct = self.get_secondary_structure(raw_embedding)
        subcell = self.get_subcellular_location(raw_embedding)

        return BasicExtractedAnnotations(disorder=secstruct.disorder, DSSP8=secstruct.DSSP8,
                                         DSSP3=secstruct.DSSP3, localization=subcell.localization,
                                         membrane=subcell.membrane)

    @staticmethod
    def _class2label(label_dict, yhat) -> List[Enum]:
        # get index of output node with max. activation (=predicted class)
        class_indices = torch.max(yhat, dim=1)[1].squeeze()
        return [label_dict[class_idx.item()] for class_idx in class_indices]

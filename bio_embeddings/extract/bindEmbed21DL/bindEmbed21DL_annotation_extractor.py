import logging
import numpy

import torch
import collections

from typing import List, Union
from numpy import ndarray
from enum import Enum

from bio_embeddings.extract.annotations import BindingResidues
from bio_embeddings.extract.bindEmbed21DL.binding_residues_cnn import BindingResiduesCNN
from bio_embeddings.utilities import get_device, get_model_file

logger = logging.getLogger(__name__)

# Label mappings
_metal_labels = {
    0: BindingResidues.non_binding,
    1: BindingResidues.metal
}

_nucleic_labels = {
    0: BindingResidues.non_binding,
    1: BindingResidues.nucleic_acid
}

_small_molecules_labels = {
    0: BindingResidues.non_binding,
    1: BindingResidues.small_molecule
}


BasicBindingResidueResult = collections.namedtuple('BasicBindingResidueResult',
                                                   'metal_ion nucleic_acids small_molecules')


class BindEmbed21DLAnnotationExtractor:
    """
    Extract binding predictions for 3 different ligand classes (metal ions, nucleic acids, small molecules).
    Residues are considered as binding to a specific class if the output probability is >= 0.5 for this class.
    """
    necessary_files = ["model_1_file", "model_2_file", "model_3_file", "model_4_file", "model_5_file"]

    def __init__(self, device: Union[None, str, torch.device] = None, **kwargs):
        """
        Initialize annotation extractor. Must define non-positional arguments for paths of files.

        :param model_file: path of bindEmbed21DL inference model checkpoint file
        """

        self._options = kwargs
        self._device = get_device(device)

        # Create un-trained (raw) models
        self._binding_residue_model_1 = BindingResiduesCNN().to(self._device)
        self._binding_residue_model_2 = BindingResiduesCNN().to(self._device)
        self._binding_residue_model_3 = BindingResiduesCNN().to(self._device)
        self._binding_residue_model_4 = BindingResiduesCNN().to(self._device)
        self._binding_residue_model_5 = BindingResiduesCNN().to(self._device)

        # Download the checkpoint files if needed
        for file in self.necessary_files:
            if not self._options.get(file):
                self._options[file] = get_model_file(model=f"bindembed21dl", file=file)

        self.model_file_1 = self._options['model_1_file']
        self.model_file_2 = self._options['model_2_file']
        self.model_file_3 = self._options['model_3_file']
        self.model_file_4 = self._options['model_4_file']
        self.model_file_5 = self._options['model_5_file']

        # load pre-trained weights for annotation machines
        binding_residue_state_1 = torch.load(self.model_file_1, map_location=self._device)
        binding_residue_state_2 = torch.load(self.model_file_1, map_location=self._device)
        binding_residue_state_3 = torch.load(self.model_file_1, map_location=self._device)
        binding_residue_state_4 = torch.load(self.model_file_1, map_location=self._device)
        binding_residue_state_5 = torch.load(self.model_file_1, map_location=self._device)

        # load pre-trained weights into raw model
        self._binding_residue_model_1.load_state_dict(binding_residue_state_1['state_dict'])
        self._binding_residue_model_2.load_state_dict(binding_residue_state_2['state_dict'])
        self._binding_residue_model_3.load_state_dict(binding_residue_state_3['state_dict'])
        self._binding_residue_model_4.load_state_dict(binding_residue_state_4['state_dict'])
        self._binding_residue_model_5.load_state_dict(binding_residue_state_5['state_dict'])

        # ensure that model is in evaluation mode (important for batchnorm and dropout)
        self._binding_residue_model_1.eval()
        self._binding_residue_model_2.eval()
        self._binding_residue_model_3.eval()
        self._binding_residue_model_4.eval()
        self._binding_residue_model_5.eval()

    def get_binding_residues(self, raw_embedding: ndarray) -> BasicBindingResidueResult:
        sigm = torch.nn.Sigmoid()

        raw_embedding = raw_embedding.astype(numpy.float32)  # For T5 fp16
        embedding = torch.tensor(raw_embedding).to(self._device)

        # Pass embeddings to the 5 different models to produce predictions
        pred_binding_1 = sigm(self._binding_residue_model_1(embedding))
        pred_binding_2 = sigm(self._binding_residue_model_2(embedding))
        pred_binding_3 = sigm(self._binding_residue_model_3(embedding))
        pred_binding_4 = sigm(self._binding_residue_model_4(embedding))
        pred_binding_5 = sigm(self._binding_residue_model_5(embedding))

        pred_binding_1 = torch.round(pred_binding_1 * 10 ** 3) / 10 ** 3
        pred_binding_2 = torch.round(pred_binding_2 * 10 ** 3) / 10 ** 3
        pred_binding_3 = torch.round(pred_binding_3 * 10 ** 3) / 10 ** 3
        pred_binding_4 = torch.round(pred_binding_4 * 10 ** 3) / 10 ** 3
        pred_binding_5 = torch.round(pred_binding_5 * 10 ** 3) / 10 ** 3

        pred_binding = (pred_binding_1 + pred_binding_2 + pred_binding_3 + pred_binding_4 + pred_binding_5) / 5

        # Map raw class predictions (integers) to class labels (strings)
        pred_metal = self._class2label(_metal_labels, pred_binding[0, ])
        pred_nuc = self._class2label(_nucleic_labels, pred_binding[1, ])
        pred_small = self._class2label(_small_molecules_labels, pred_binding[2, ])

        return BasicBindingResidueResult(metal_ion=pred_metal, nucleic_acids=pred_nuc, small_molecules=pred_small)

    @staticmethod
    def _class2label(label_dict, yhat) -> List[Enum]:
        # get index of output node with max. activation (=predicted class)
        class_indices = torch.ge(yhat, 0.5).int()
        return [label_dict[class_idx.item()] for class_idx in class_indices]

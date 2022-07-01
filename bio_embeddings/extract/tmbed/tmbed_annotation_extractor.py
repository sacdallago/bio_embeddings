import logging

import torch
import numpy
import collections

from typing import List, Union

from bio_embeddings.extract.annotations import MembraneResidues
from bio_embeddings.extract.tmbed.tmbed_cnn import TmbedModel
from bio_embeddings.extract.tmbed.tmbed_viterbi import Decoder
from bio_embeddings.utilities import get_device, get_model_file


logger = logging.getLogger(__name__)


# Label mappings
_tmbed_labels = {
    0: MembraneResidues.TMB_IN_OUT,
    1: MembraneResidues.TMB_OUT_IN,
    2: MembraneResidues.TMH_IN_OUT,
    3: MembraneResidues.TMH_OUT_IN,
    4: MembraneResidues.SIGNAL_PEPTIDE,
    5: MembraneResidues.NON_TRANSMEMBRANE,
    6: MembraneResidues.NON_TRANSMEMBRANE,
}


MembraneResiduesResult = collections.namedtuple(
    'MembraneResiduesResult', 'membrane_residues'
)


def make_mask(embeddings, lengths):
    B, N, _ = embeddings.shape

    mask = torch.zeros((B, N),
                       dtype=embeddings.dtype,
                       device=embeddings.device)

    for idx, length in enumerate(lengths):
        mask[idx, :length] = 1.0

    return mask


class TmbedAnnotationExtractor:
    '''
    Extract membrane residue predictions using 4 different classes:
        - Transmembrane beta strand
        - Transmembrane alpha helix
        - Signal Peptide
        - Non-Transmembrane
    '''

    necessary_files = ['model_0_file',
                       'model_1_file',
                       'model_2_file',
                       'model_3_file',
                       'model_4_file']

    def __init__(self, device: Union[None, str, torch.device] = None, **kwargs):

        self._options = kwargs
        self._device = get_device(device)

        # Download the checkpoint files if needed
        for file in self.necessary_files:
            if not self._options.get(file):
                self._options[file] = get_model_file(model='tmbed', file=file)

        self._models = []

        for model_idx in range(5):
            # Create blank model
            model = TmbedModel()
            # Get model file
            model_file = self._options[f'model_{model_idx}_file']
            # Load pre-trained weights
            model.load_state_dict(torch.load(model_file)['model'])
            # Finalize model
            model = model.eval().to(self._device)
            # Add to model list
            self._models.append(model)

        self._decoder = Decoder()

    def get_membrane_residues(self, raw_embedding: numpy.ndarray, lengths: List[int]) -> List[MembraneResiduesResult]:
        # Make torch.float32 embeddings and move to device
        embeddings = torch.from_numpy(raw_embedding).to(self._device).float()

        B, N, _ = embeddings.shape

        assert B == len(lengths)

        # Prepare mask and prediction tensor
        mask = make_mask(embeddings, lengths)
        prediction = torch.zeros((B, 5, N), device=embeddings.device)

        # Predict average class labels
        with torch.no_grad():
            for model in self._models:
                y = model(embeddings, mask)
                prediction = prediction + torch.softmax(y, dim=1)

            prediction = prediction / len(self._models)

        # Decode class labels
        mask = mask.cpu()
        prediction = prediction.cpu()
        prediction = self._decoder(prediction, mask).byte()

        # Finalize predictions
        final_predictions = []

        for idx, length in enumerate(lengths):
            pred = prediction[idx, :length]
            pred = [_tmbed_labels[label] for label in pred.tolist()]
            pred = MembraneResiduesResult(membrane_residues=pred)

            final_predictions.append(pred)

        return final_predictions

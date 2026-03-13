from pathlib import Path
from typing import Union

import torch
from numpy import ndarray
from torch import nn


class PBTuckerModel(nn.Module):
    """This is the torch module behind :class:`PBTucker`"""
    def __init__(self):
        super(PBTuckerModel, self).__init__()
        self.tucker = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
        )

    def forward(self, data: torch.tensor) -> torch.tensor:
        return self.tucker(data)


class PBTucker:
    """Tucker is a contrastive learning model trained to distinguish CATH superfamilies.

    It consumes prottrans_bert_bfd embeddings and reduces the embedding dimensionality from 1024 to 128.
    See https://www.biorxiv.org/content/10.1101/2021.01.21.427551v1

    To use it outside of the pipeline, first instantiate it with
    `pb_tucker = PBTucker("/path/to/model", device)`,
    then project your reduced bert embedding with
    `pb_tucker.project_reduced_embedding(bert_embedding)`.
    """

    _device: torch.device
    name: str = "pb_tucker"

    def __init__(self, model_file: Union[str, Path], device: torch.device, n_components: int):
        self._device = device
        self.model = PBTuckerModel()
        self.model.load_state_dict(
            torch.load(model_file, map_location=device)["state_dict"]
        )
        self.model.eval()
        self.model = self.model.to(self._device)
        self.n_components = n_components

    def project_reduced_embedding(self, reduced_embedding: ndarray) -> ndarray:
        with torch.no_grad():
            reduced_embedding_tensor = torch.tensor(
                reduced_embedding, device=self._device
            )
            return self.model.tucker(reduced_embedding_tensor).cpu().numpy()

    def fit_transform(self, embeddings: ndarray) -> ndarray:
        return [arr[:self.n_components] for arr in embeddings]

def pb_tucker_reduce(embeddings, **kwargs):
    """Wrapper around :meth:`bio_embeddings.project.PBTucker` with defaults for bio_embeddings"""
    pb_tucker_params = dict()

    pb_tucker_params['n_components'] = kwargs.get('n_components', 3)
    pb_tucker_params['model_file'] = kwargs.get('model_file', model_file)
    pb_tucker_params['device'] = kwargs.get('device_object', device)

    transformed_embeddings = PBTucker(**pb_tucker_params).fit_transform(embeddings)

    return transformed_embeddings

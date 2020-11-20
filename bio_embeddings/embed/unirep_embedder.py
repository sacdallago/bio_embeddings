from typing import Any, Dict, Union

import torch
from numpy import ndarray

from bio_embeddings.embed import EmbedderInterface


class UniRepEmbedder(EmbedderInterface):
    """UniRep Embedder

    Alley, E.C., Khimulya, G., Biswas, S. et al. Unified rational protein
    engineering with sequence-based deep representation learning. Nat Methods
    16, 1315–1322 (2019). https://doi.org/10.1038/s41592-019-0598-1

    We use a reimplementation of unirep:

    Ma, Eric, and Arkadij Kummer. "Reimplementing Unirep in JAX." bioRxiv (2020).
    https://doi.org/10.1101/2020.05.11.088344
    """

    name = "unirep"
    # An integer representing the size of the embedding.
    embedding_dimension = 1900
    # An integer representing the number of layers from the RAW output of the LM.
    number_of_layers = 1

    params: Dict[str, Any]

    def __init__(self, device: Union[None, str, torch.device] = None, **kwargs):
        from jax_unirep.utils import load_params_1900

        if device:
            raise NotImplementedError("UniRep does not allow configuring the device")
        super().__init__(device, **kwargs)

        self.params = load_params_1900()

    def embed(self, sequence: str) -> ndarray:
        from jax import vmap, partial
        from jax_unirep.featurize import apply_fun
        from jax_unirep.utils import get_embeddings

        # Unirep only allows batching with sequences of the same length, so we don't do batching at all
        embedded_seqs = get_embeddings([sequence])
        # h and c refer to hidden and cell state
        # h contains all the hidden states, while h_final and c_final contain only the last state
        h_final, c_final, h = vmap(partial(apply_fun, self.params))(embedded_seqs)
        # Go from a batch of 1, which is `(1, len(sequence), 1900)`, to `len(sequence), 1900)`
        return h[0]

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        # This is `h_avg` in jax-unirep terminology
        return embedding.mean(axis=0)

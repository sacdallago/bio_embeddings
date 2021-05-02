from typing import Any, Dict, Union, Callable

import numpy
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

    _params: Dict[str, Any]
    _apply_fun: Callable

    def __init__(self, device: Union[None, str, torch.device] = None, **kwargs):
        from jax_unirep.utils import load_params_1900
        from jax_unirep.featurize import apply_fun

        self._params = load_params_1900()
        self._apply_fun = apply_fun

        # For v2
        # https://github.com/ElArkk/jax-unirep/issues/107
        # from jax_unirep.utils import load_params
        # from jax_unirep.layers import mLSTM
        # from jax_unirep.utils import validate_mLSTM_params
        # self._params = load_params()[1]
        # _, self._apply_fun = mLSTM(output_dim=self.embedding_dimension)
        # validate_mLSTM_params(self._params, n_outputs=self.embedding_dimension)

        if device:
            raise NotImplementedError("UniRep does not allow configuring the device")
        super().__init__(device, **kwargs)

    def embed(self, sequence: str) -> ndarray:
        from jax import vmap, partial
        from jax_unirep.utils import get_embeddings

        # https://github.com/sacdallago/bio_embeddings/issues/117
        if not sequence:
            return numpy.zeros((0, self.embedding_dimension))

        # Unirep only allows batching with sequences of the same length, so we don't do batching at all
        embedded_seqs = get_embeddings([sequence])
        # h and c refer to hidden and cell state
        # h contains all the hidden states, while h_final and c_final contain only the last state
        h_final, c_final, h = vmap(partial(self._apply_fun, self._params))(
            embedded_seqs
        )
        # Go from a batch of 1, which is `(1, len(sequence), 1900)`, to `len(sequence), 1900)`
        return numpy.asarray(h[0])

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        # This is `h_avg` in jax-unirep terminology
        return embedding.mean(axis=0)

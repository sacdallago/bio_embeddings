from typing import List, Generator, Union

import torch
from esm.pretrained import load_model_and_alphabet_local
from numpy import ndarray

from bio_embeddings.embed import EmbedderInterface


class ESMEmbedderBase(EmbedderInterface):
    # The only thing we need to overwrite is the name and _picked_layer
    embedding_dimension = 1280
    number_of_layers = 1  # Following ESM, we only consider layer 34 (ESM) or 33 (ESM1b)

    _necessary_files = ["model_file"]

    _picked_layer: int

    def __init__(self, device: Union[None, str, torch.device] = None, **kwargs):
        super().__init__(device, **kwargs)

        model, alphabet = load_model_and_alphabet_local(self._options["model_file"])

        self._model = model.to(self._device)
        self._batch_converter = alphabet.get_batch_converter()

    def embed(self, sequence: str) -> ndarray:
        [embedding] = self.embed_batch([sequence])
        return embedding

    def embed_batch(self, batch: List[str]) -> Generator[ndarray, None, None]:
        """https://github.com/facebookresearch/esm/blob/dfa524df54f91ef45b3919a00aaa9c33f3356085/README.md#quick-start-"""
        data = [(str(pos), sequence) for pos, sequence in enumerate(batch)]
        batch_labels, batch_strs, batch_tokens = self._batch_converter(data)

        with torch.no_grad():
            results = self._model(
                batch_tokens.to(self._device), repr_layers=[self._picked_layer]
            )
        token_embeddings = results["representations"][self._picked_layer]

        # Generate per-sequence embeddings via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        for i, (_, seq) in enumerate(data):
            yield token_embeddings[i, 1 : len(seq) + 1].cpu().numpy()

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        return embedding.mean(0)


class ESMEmbedder(ESMEmbedderBase):
    """ESM Embedder (Note: This is not ESM-1b)

    Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences

    Rives, Alexander, et al. "Biological structure and function emerge from
    scaling unsupervised learning to 250 million protein sequences."
    bioRxiv (2019): 622803. https://doi.org/10.1101/622803
    """

    name = "esm"
    _picked_layer = 34


class ESM1bEmbedder(ESMEmbedderBase):
    """ESM-1b Embedder (Note: This is not the original ESM)

    Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences

    Rives, Alexander, et al. "Biological structure and function emerge from
    scaling unsupervised learning to 250 million protein sequences."
    bioRxiv (2019): 622803. https://doi.org/10.1101/622803
    """

    name = "esm1b"
    _picked_layer = 33

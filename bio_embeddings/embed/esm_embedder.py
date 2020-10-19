from typing import List, Generator, Union

import esm
import torch
from numpy import ndarray

from bio_embeddings.embed import EmbedderInterface


class ESMEmbedder(EmbedderInterface):
    """ESM Embedder
    Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences

    Alexander Rives, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Demi Guo, Myle Ott, C. Lawrence Zitnick, Jerry Ma, Rob Fergus
    bioRxiv 622803; doi: https://doi.org/10.1101/622803

    Idle GPU usage: 4134MiB
    """

    name = "esm"
    embedding_dimension = 1280
    number_of_layers = 1  # Following ESM, we only consider layer 34

    _necessary_files = ["model_file"]

    def __init__(self, device: Union[None, str, torch.device] = None, **kwargs):
        super().__init__(device, **kwargs)
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(
            self._options["model_file"]
        )
        self._model = model.to(self._device)
        self._batch_converter = alphabet.get_batch_converter()

    def embed(self, sequence: str) -> ndarray:
        [embedding] = self.embed_batch([sequence])
        return embedding

    def embed_batch(self, batch: List[str]) -> Generator[ndarray, None, None]:
        """See https://github.com/facebookresearch/esm/blob/master/README.rst#quickstart"""
        data = [(str(pos), sequence) for pos, sequence in enumerate(batch)]
        batch_labels, batch_strs, batch_tokens = self._batch_converter(data)

        with torch.no_grad():
            results = self._model(batch_tokens.to(self._device), repr_layers=[34])
        token_embeddings = results["representations"][34]

        # Generate per-sequence embeddings via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        for i, (_, seq) in enumerate(data):
            yield token_embeddings[i, 1: len(seq) + 1].cpu().numpy()

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        return embedding.mean(0)

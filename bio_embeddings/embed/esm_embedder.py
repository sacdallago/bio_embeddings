from argparse import Namespace
from typing import List, Generator, Union

import esm
import torch
from esm.constants import proteinseq_toks
from numpy import ndarray

from bio_embeddings.embed import EmbedderInterface


class ESMEmbedder(EmbedderInterface):
    """ESM Embedder

    Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences

    Rives, Alexander, et al. "Biological structure and function emerge from
    scaling unsupervised learning to 250 million protein sequences."
    bioRxiv (2019): 622803. https://doi.org/10.1101/622803
    """

    name = "esm"
    embedding_dimension = 1280
    number_of_layers = 1  # Following ESM, we only consider layer 34

    _necessary_files = ["model_file"]

    def __init__(self, device: Union[None, str, torch.device] = None, **kwargs):
        super().__init__(device, **kwargs)

        alphabet = esm.Alphabet.from_dict(proteinseq_toks)
        if torch.cuda.is_available():
            model_data = torch.load(self._options["model_file"])
        else:
            model_data = torch.load(self._options["model_file"], map_location=torch.device('cpu'))

        # upgrade state dict
        pra = lambda s: ''.join(s.split('decoder_')[1:] if 'decoder' in s else s)
        prs = lambda s: ''.join(s.split('decoder.')[1:] if 'decoder' in s else s)
        model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
        model_state = {prs(arg[0]): arg[1] for arg in model_data["model"].items()}
        model = esm.ProteinBertModel(
            Namespace(**model_args), len(alphabet), padding_idx=alphabet.padding_idx
        )
        model.load_state_dict(model_state)

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

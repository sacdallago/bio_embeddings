from typing import List, Generator, Union

import numpy
import torch
from CPCProt import CPCProtModel, CPCProtEmbedding
from CPCProt.collate_fn import pad_sequences
from CPCProt.tokenizer import Tokenizer
from numpy import ndarray

from bio_embeddings.embed import EmbedderInterface


class CPCProtEmbedder(EmbedderInterface):
    """CPCProt Embedder
    Self-Supervised Contrastive Learning of Protein Representations By Mutual Information Maximization
    Amy X. Lu, Haoran Zhang, Marzyeh Ghassemi, Alan Moses
    bioRxiv 2020.09.04.283929; doi: https://doi.org/10.1101/2020.09.04.283929
    """

    name = "cpcprot"
    embedding_dimension = NotImplementedError
    number_of_layers = 1

    _necessary_files = ["model_file"]

    def __init__(self, device: Union[None, str, torch.device] = None, **kwargs):
        super().__init__(device, **kwargs)
        self.tokenizer = Tokenizer(vocab="iupac")
        raw_model = CPCProtModel().to(self._device)
        state_dict = dict(
            torch.load(self._options["model_file"], map_location=self._device)
        )
        for i in list(state_dict.keys()):
            if i.startswith("module."):
                state_dict[i[7:]] = state_dict[i]
                del state_dict[i]
        raw_model.load_state_dict(state_dict)
        self._model = CPCProtEmbedding(raw_model.to(self._device).eval())

    def embed(self, sequence: str) -> ndarray:
        [embedding] = self.embed_batch([sequence])
        return embedding

    def embed_batch(self, batch: List[str]) -> Generator[ndarray, None, None]:
        """See https://github.com/amyxlu/CPCProt/blob/df1ad1118544ed349b5e711207660a7c205b3128/embed_fasta.py"""
        encoded = [numpy.array(self.tokenizer.encode(sequence)) for sequence in batch]
        torch_inputs = torch.from_numpy(pad_sequences(encoded, 0))
        yield from self._model.get_z_mean(torch_inputs).detach().cpu().numpy()

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        return embedding

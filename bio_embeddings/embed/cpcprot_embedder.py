from typing import List, Generator, Union

import numpy
import torch
from CPCProt import CPCProtModel, CPCProtEmbedding, CPCProtConfig
from CPCProt.model.cpcprot import DEFAULT_CONFIG
from CPCProt.tokenizer import Tokenizer
from numpy import ndarray

from bio_embeddings.embed import EmbedderInterface


class CPCProtEmbedder(EmbedderInterface):
    """CPCProt Embedder

    Lu, Amy X., et al. "Self-supervised contrastive learning of protein
    representations by mutual information maximization." bioRxiv (2020).
    https://doi.org/10.1101/2020.09.04.283929
    """

    name = "cpcprot"
    embedding_dimension = 512
    number_of_layers = 1

    necessary_files = ["model_file"]

    def __init__(self, device: Union[None, str, torch.device] = None, **kwargs):
        super().__init__(device, **kwargs)
        self.tokenizer = Tokenizer(vocab="iupac")
        # If we don't do this here, CPCProtModel will end up on the gpu if one is
        # available, even if we passed the cpu as device.
        # Afaik this is the best way to derive from DEFAULT_CONFIG
        dict_cfg = DEFAULT_CONFIG.to_dict()
        dict_cfg["use_cuda"] = self._device.type == "cuda"
        raw_model = CPCProtModel(cfg=CPCProtConfig.from_dict(dict_cfg)).to(self._device)
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
        # 11 is the minimum patch size, so we need to zero-pad shorter sequences
        pad_length = max(max([i.shape[0] for i in encoded]), 11)
        padded = [numpy.pad(i, (0, pad_length - i.shape[0])) for i in encoded]
        torch_inputs = torch.from_numpy(numpy.array(padded))
        yield from self._model.get_z_mean(torch_inputs).detach().cpu().numpy()

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        return embedding

import warnings
from typing import List, Generator, Union, Iterable, Optional, Any, Tuple
from itertools import tee

import torch
from esm.pretrained import load_model_and_alphabet_core
from numpy import ndarray

from bio_embeddings.embed import EmbedderInterface
from bio_embeddings.utilities import get_model_file


def load_model_and_alphabet_local(model_location: str) -> Tuple[Any, Any]:
    """Custom bio_embeddings versions because we change names and don't have regression weights"""
    # We don't predict contacts
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Regression weights not found, predicting contacts will not produce correct results.",
    )

    model_data = torch.load(model_location, map_location="cpu")
    return load_model_and_alphabet_core(model_data, None)


class ESMEmbedderBase(EmbedderInterface):
    # The only thing we need to overwrite is the name and _picked_layer
    embedding_dimension = 1280
    number_of_layers = 1  # Following ESM, we only consider layer 34 (ESM) or 33 (ESM1b)
    necessary_files = ["model_file"]
    # https://github.com/facebookresearch/esm/issues/49#issuecomment-803110092
    max_len = 1022

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
        batch, batch_copy = tee(batch)
        self._assert_max_len(batch_copy)
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

    def embed_many(
        self, sequences: Iterable[str], batch_size: Optional[int] = None
    ) -> Generator[ndarray, None, None]:
        sequences, sequences_copy = tee(sequences)
        self._assert_max_len(sequences_copy)
        yield from super().embed_many(sequences, batch_size)

    def _assert_max_len(self, sequences: Iterable[str]):
        max_len = max((len(i) for i in sequences), default=0)
        if max_len > self.max_len:
            raise ValueError(
                f"{self.name} only allows sequences up to {self.max_len} residues, "
                f"but your longest sequence is {max_len} residues long"
            )

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        return embedding.mean(0)


class ESMEmbedder(ESMEmbedderBase):
    """ESM Embedder (Note: This is not ESM-1b)

    Rives, Alexander, et al. "Biological structure and function emerge from scaling unsupervised learning to 250 million
    protein sequences." Proceedings of the National Academy of Sciences 118.15 (2021).
    https://doi.org/10.1073/pnas.2016239118
    """

    name = "esm"
    _picked_layer = 34


class ESM1bEmbedder(ESMEmbedderBase):
    """ESM-1b Embedder (Note: This is not the original ESM)

    Rives, Alexander, et al. "Biological structure and function emerge from scaling unsupervised learning to 250 million
    protein sequences." Proceedings of the National Academy of Sciences 118.15 (2021).
    https://doi.org/10.1073/pnas.2016239118
    """

    name = "esm1b"
    _picked_layer = 33


class ESM1vEmbedder(ESMEmbedderBase):
    """ESM-1v Embedder (one of five)

    ESM1v uses an ensemble of five models, called `esm1v_t33_650M_UR90S_[1-5]`. An instance of this class is one
    of the five, specified by `ensemble_id`.

    Meier, Joshua, et al. "Language models enable zero-shot prediction of the effects of mutations on protein function."
    bioRxiv (2021). https://doi.org/10.1101/2021.07.09.450648
    """

    name = "esm1v"
    ensemble_id: int
    _picked_layer = 33

    def __init__(
        self, ensemble_id: int, device: Union[None, str, torch.device] = None, **kwargs
    ):
        """You must pass the number of the model (1-5) as first parameter, though you can override the weights file with
        model_file"""
        assert ensemble_id in range(1, 6), "The model number must be in 1-5"
        self.ensemble_id = ensemble_id

        # EmbedderInterface assumes static model files, but we need to dynamically select one of the five
        if "model_file" not in kwargs:
            kwargs["model_file"] = get_model_file(
                model=self.name, file=f"model_{ensemble_id}_file"
            )

        super().__init__(device, **kwargs)

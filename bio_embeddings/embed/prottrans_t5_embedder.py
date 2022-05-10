import abc
import logging
import os.path
import re
from itertools import zip_longest
from typing import List, Generator, Union

import torch
import transformers
from numpy import ndarray
from transformers import T5Tokenizer, T5Model, T5EncoderModel

from bio_embeddings.embed.embedder_interfaces import EmbedderWithFallback

logger = logging.getLogger(__name__)


class FilterT5DecoderWeightsWarning(logging.Filter):
    """transformers complains at length that we pass decoder weights when initializing only the encoder,
    which we can ignore"""

    def filter(self, record: logging.LogRecord) -> bool:
        return (
                "were not used when initializing T5EncoderModel: ['decoder."
                not in record.getMessage()
        )


transformers.modeling_utils.logger.addFilter(FilterT5DecoderWeightsWarning())


class ProtTransT5Embedder(EmbedderWithFallback, abc.ABC):
    """Encoder of the ProtTrans T5 model, both BFD and BFD finetuned on UniRef50. To embed please pick either
    ProtTransT5BFDEmbedder or ProtTransT5UniRef50Embedder

    Note that this model alone takes 13GB, so you need a GPU with a lot of memory.
    """

    _model: Union[T5Model, T5EncoderModel]
    _decoder: bool = False
    _half_precision_model: bool = False
    embedding_dimension = 1024
    number_of_layers = 1
    necessary_directories = ["model_directory"]

    def __init__(self, **kwargs):
        """
        Initialize T5 embedder.

        :param str model_directory: where the weights of the model can be found
        :param device: whether to compute on the CPU or GPU
        :type device: str or torch.device or None
        :param bool decoder: Whether to use also the decoder (default: False)
        :param bool half_precision_model: Use the model in half precision (float16) mode (default: False)
        """
        # HIWI Benjamin
        # The user can use the half precision model either by specifiing the path or setting the flag
        # The half precision model with be used if either the flag is set or a path providied
        # This is performed before calling super so that the paths can be fetched if not provided
        if ('half_precision_model' in kwargs.keys() or 'half_precision_model_directory' in kwargs.keys()) and 'model_directory' not in kwargs.keys():
            # the necessary directories are changed since now 'model_directory' isn't needed but 'half_precision_model_dir' is
            self.necessary_directories = ["half_precision_model_directory"]
            # if the path was provided and the flag wasn't this sets the flag for later use
            kwargs['half_precision_model'] = True

        super().__init__(**kwargs)

        # set the model directory depending on whether to use half precision
        if 'half_precision_model' in kwargs.keys() and 'model_directory' not in kwargs.keys():
            self._model_directory = self._options["half_precision_model_directory"]
        else:
            self._model_directory = self._options["model_directory"]


        # Until we know whether we need the decoder, let's keep it here as an undocumented option.
        # Should the need arise we can just split this class in to an encoder and a decoder subclass
        # by setting one subclass to _decoder=True and the other to _decoder=False
        self._decoder = self._options.get("decoder", False)
        self._half_precision_model = self._options.get("half_precision_model", False)

        self._model = self.get_model().to(self._device).eval()
        self._model_fallback = None
        self._tokenizer = T5Tokenizer.from_pretrained(
            self._model_directory, do_lower_case=False
        )

    def get_model(self) -> Union[T5Model, T5EncoderModel]:

        if not self._decoder:
            if self._half_precision_model:
                model = T5EncoderModel.from_pretrained(self._model_directory, torch_dtype=torch.float16)
            else:
                model = T5EncoderModel.from_pretrained(self._model_directory)
        else:
            if self._half_precision_model:
                model = T5Model.from_pretrained(self._model_directory, torch_dtype=torch.float16)
            else:
                model = T5Model.from_pretrained(self._model_directory)

        return model

    def _get_fallback_model(self) -> Union[T5Model, T5EncoderModel]:
        """ Returns the CPU model """
        if self._half_precision_model:
            raise NotImplementedError(
                "You sequence was too long for the GPU, "
                "but we can't fall back to the CPU with half_precision_model=True "
                "(https://github.com/huggingface/transformers/issues/11546)"
            )
        if not self._model_fallback:
            self._model_fallback = self.get_model()
        return self._model_fallback

    def _embed_batch_impl(
            self, batch: List[str], model: T5Model
    ) -> Generator[ndarray, None, None]:
        seq_lens = [len(seq) for seq in batch]
        # Remove rare amino acids
        batch = [re.sub(r"[UZOB]", "X", sequence) for sequence in batch]
        # transformers needs spaces between the amino acids
        batch = [" ".join(list(seq)) for seq in batch]

        if not batch:
            return

        ids = self._tokenizer.batch_encode_plus(
            batch, add_special_tokens=True, padding="longest"
        )

        tokenized_sequences = torch.tensor(ids["input_ids"]).to(model.device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(model.device)

        with torch.no_grad():
            if not self._decoder:
                embeddings = model(
                    input_ids=tokenized_sequences, attention_mask=attention_mask
                )
            else:
                embeddings = model(
                    input_ids=tokenized_sequences,
                    attention_mask=attention_mask,
                    decoder_input_ids=tokenized_sequences,
                )

        embeddings = embeddings[0].cpu().numpy()

        for seq_num, seq_len in zip_longest(range(len(embeddings)), seq_lens):
            # slice off last position (special token)
            embedding = embeddings[seq_num][:seq_len]
            assert (
                    seq_len == embedding.shape[0]
            ), f"Sequence length mismatch: {seq_len} vs {embedding.shape[0]}"

            yield embedding

    @staticmethod
    def reduce_per_protein(embedding):
        return embedding.mean(axis=0)

    def embed(self, sequence: str) -> ndarray:
        [embedding] = self.embed_batch([sequence])
        return embedding


class ProtTransT5BFDEmbedder(ProtTransT5Embedder):
    """Encoder of the ProtTrans T5 model trained on BFD.
    Consider using :class:`ProtTransT5XLU50Embedder` instead of this

    We recommend settings `half_model=True`, which on the tested GPU (Quadro RTX 3000) reduces memory consumption
    from 12GB to 7GB while the effect in benchmarks is negligible (±0.1 percentages points in different sets,
    generally below standard error)
    """

    name = "prottrans_t5_bfd"


class ProtTransT5UniRef50Embedder(ProtTransT5Embedder):
    """Encoder of the ProtTrans T5 model trained on BFD and finetuned on UniRef 50.
    Consider using :class:`ProtTransT5XLU50Embedder` instead of this

    We recommend settings `half_model=True`, which on the tested GPU (Quadro RTX 3000) reduces memory consumption
    from 12GB to 7GB while the effect in benchmarks is negligible (±0.1 percentages points in different sets,
    generally below standard error)
    """

    name = "prottrans_t5_uniref50"


class ProtTransT5XLU50Embedder(ProtTransT5Embedder):
    """Encoder of the ProtTrans T5 model trained on BFD and finetuned on UniRef 50.

    We recommend settings `half_model=True`, which on the tested GPU (Quadro RTX 3000) reduces memory consumption
    from 12GB to 7GB while the effect in benchmarks is negligible (±0.1 percentages points in different sets,
    generally below standard error)
    """

    name = "prottrans_t5_xl_u50"

import logging
import re
from typing import Union, Optional, List, Dict

import torch
import transformers
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM

from bio_embeddings.embed import ProtTransBertBFDEmbedder
from bio_embeddings.mutagenesis import AMINO_ACIDS
from bio_embeddings.utilities import (
    get_device,
    get_model_directories_from_zip,
)


class FilterBertForMaskedLMWeightsWarning(logging.Filter):
    """transformers complains that we don't use some of the weights with BertForMaskedLM instead of BertModel,
    which we can ignore"""

    def filter(self, record: logging.LogRecord) -> bool:
        return (
            "were not used when initializing BertForMaskedLM: "
            "['cls.seq_relationship.weight', 'cls.seq_relationship.bias']"
            not in record.getMessage()
        )


transformers.modeling_utils.logger.addFilter(FilterBertForMaskedLMWeightsWarning())


class ProtTransBertBFDMutagenesis:
    """BETA: in-silico mutagenesis using BertForMaskedLM"""

    device: torch.device
    model: BertForMaskedLM
    tokenizer: BertTokenizer
    _half_precision_model: bool

    def __init__(
        self,
        device: Union[None, str, torch.device] = None,
        model_directory: Optional[str] = None,
        half_precision_model: bool = False,
    ):
        """Loads the Bert Model for Masked LM"""
        self.device = get_device(device)
        self._half_precision_model = half_precision_model

        if not model_directory:
            model_directory = get_model_directories_from_zip(
                model=ProtTransBertBFDEmbedder.name, directory="model_directory"
            )
        self.tokenizer = BertTokenizer.from_pretrained(
            model_directory, do_lower_case=False
        )
        self.model = BertForMaskedLM.from_pretrained(model_directory)
        # Compute in half precision, which is a lot faster and saves us half the memory
        if self._half_precision_model:
            self.model = self.model.half()
        self.model = self.model.eval().to(self.device)

    def get_sequence_probabilities(
        self,
        sequence: str,
        temperature: float = 1,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        progress_bar: Optional[tqdm] = None,
    ) -> List[Dict[str, float]]:
        """Returns the likelihood for each of the 20 natural amino acids to be at residue positions between `start` and
        `end` considering the context of the remainder of the sequence (aka: by using. BERT's mask token and
        reconstructing the corrupted sequence). Probabilities may be adjusted by a `temperature` factor.
        If set to `1` (default) no adjustment is made.

        :param sequence: The amino acid sequence. Please pass whole sequences, not regions
        :param start: the start index (inclusive) of the region for which to compute residue probabilities (starting with 0)
        :param stop: the end (exclusive) of the region for which to compute residue probabilities
        :param temperature: temperature for the softmax computation
        :param progress_bar: optional tqdm progress bar

        :return: An ordered list for the region of probabilities for each of the 20 natural amino acids to be at said
        position."""
        # https://stackoverflow.com/questions/59435020/get-probability-of-multi-token-word-in-mask-position

        # init softmax to get mutagenesis later on
        sm = torch.nn.Softmax(dim=0)
        AA_tokens = [
            self.tokenizer.convert_tokens_to_ids(AA) for AA in list(AMINO_ACIDS)
        ]

        # Create L sequences with each position masked once
        probabilities_list = list()

        # Remove rare amino acids
        current_sequence = re.sub(r"[UZOB]", "X", sequence)

        # Mask each token individually
        for i in range(start or 0, stop or len(sequence)):
            masked_sequence = list(current_sequence)
            masked_sequence = (
                masked_sequence[:i]
                + [self.tokenizer.mask_token]
                + masked_sequence[i + 1 :]
            )
            # Each AA is a word, so we need spaces in between
            masked_sequence = " ".join(masked_sequence)
            tokenized_sequence = self.tokenizer.encode(
                masked_sequence, return_tensors="pt"
            )

            # get the position of the masked token
            # noinspection PyTypeChecker
            masked_position = torch.nonzero(
                tokenized_sequence.squeeze() == self.tokenizer.mask_token_id
            ).item()

            # TODO: can batch this!
            output = self.model(tokenized_sequence.to(self.device))
            last_hidden_state = output[0].squeeze(0)
            # only get output for masked token
            # output is the size of the vocabulary
            mask_hidden_state = last_hidden_state[masked_position].cpu()

            # convert to mutagenesis (softmax)
            # giving a probability for each item in the vocabulary
            probabilities = sm(mask_hidden_state / temperature)

            # Get a dictionary of AA and probability of it being there at given position
            result = dict(
                zip(list(AMINO_ACIDS), [probabilities[AA].item() for AA in AA_tokens])
            )
            result["position"] = i

            # Append orderly to mutagenesis
            probabilities_list.append(result)

            if progress_bar:
                progress_bar.update()

        return probabilities_list

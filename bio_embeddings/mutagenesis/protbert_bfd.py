import re
from typing import Union, Tuple, Any, Optional, List, Dict

import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM

from bio_embeddings.embed import ProtTransBertBFDEmbedder
from bio_embeddings.mutagenesis import AMINO_ACIDS
from bio_embeddings.utilities import (
    get_device,
    get_model_directories_from_zip,
)


def get_model(
    device: Union[None, str, torch.device] = None, model_directory: Optional[str] = None
) -> Tuple[Any, Any]:
    """Loads the Bert Model for Masked LM"""
    if not model_directory:
        model_directory = get_model_directories_from_zip(
            model=ProtTransBertBFDEmbedder.name, directory="model_directory"
        )
    # TODO: Silence "Some weights of the model checkpoint" warning
    tokenizer = BertTokenizer.from_pretrained(model_directory, do_lower_case=False)
    model = BertForMaskedLM.from_pretrained(model_directory)
    model = model.eval().to(get_device(device))
    return tokenizer, model


def get_sequence_probabilities(
    sequence: str,
    tokenizer: BertTokenizer,
    model: BertForMaskedLM,
    device: Union[None, str, torch.device] = None,
    temperature: float = 1,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    pbar: Optional[tqdm] = None,
) -> List[Dict[str, float]]:
    """Returns for each position the probabilities for each amino acid if the token is masked"""
    # https://stackoverflow.com/questions/59435020/get-probability-of-multi-token-word-in-mask-position
    device = get_device(device)

    # init softmax to get mutagenesis later on
    sm = torch.nn.Softmax(dim=0)
    AA_tokens = [tokenizer.convert_tokens_to_ids(AA) for AA in list(AMINO_ACIDS)]

    # Create L sequences with each position masked once
    probabilities_list = list()

    # Remove rare amino acids
    current_sequence = re.sub(r"[UZOB]", "X", sequence)

    # Mask each token individually
    for i in range(start or 0, stop or len(sequence)):
        masked_sequence = list(current_sequence)
        masked_sequence = (
            masked_sequence[:i] + [tokenizer.mask_token] + masked_sequence[i + 1 :]
        )
        # Each AA is a word, so we need spaces in between
        masked_sequence = " ".join(masked_sequence)
        tokenized_sequence = tokenizer.encode(masked_sequence, return_tensors="pt")

        # get the position of the masked token
        # noinspection PyTypeChecker
        masked_position = torch.nonzero(
            tokenized_sequence.squeeze() == tokenizer.mask_token_id
        ).item()

        # TODO: can batch this!
        output = model(tokenized_sequence.to(device))
        last_hidden_state = output[0].squeeze(0).cpu()

        # only get output for masked token
        # output is the size of the vocabulary
        mask_hidden_state = last_hidden_state[masked_position]

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

        if pbar:
            pbar.update()

    return probabilities_list

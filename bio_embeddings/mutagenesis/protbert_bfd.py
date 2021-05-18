import re
from typing import Union, Tuple, Any, Optional, List, Dict

import numpy
import plotly.express as px
import plotly.graph_objects as go
import torch
from pandas import DataFrame
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, PreTrainedModel

from bio_embeddings.embed import ProtTransBertBFDEmbedder
from bio_embeddings.utilities import (
    get_device,
    get_model_directories_from_zip,
)


def get_model(
    device: Union[None, str, torch.device] = None, model_directory: Optional[str] = None
) -> Tuple[Any, Any]:
    if not model_directory:
        model_directory = get_model_directories_from_zip(
            model=ProtTransBertBFDEmbedder.name, directory="model_directory"
        )
    tokenizer = BertTokenizer.from_pretrained(model_directory, do_lower_case=False)
    model = BertForMaskedLM.from_pretrained(model_directory)
    model = model.eval().to(get_device(device))
    return tokenizer, model


def get_sequence_probabilities(
    sequence: str,
    tokenizer: PreTrainedModel,
    model: PreTrainedModel,
    device: Union[None, str, torch.device] = None,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    pbar: Optional[tqdm] = None,
) -> List[Dict[str, float]]:
    """https://stackoverflow.com/questions/59435020/get-probability-of-multi-token-word-in-mask-position"""
    device = get_device(device)

    # init softmax to get mutagenesis later on
    sm = torch.nn.Softmax(dim=-1)
    AAs = "FLSYXCWPHQRIMTNKVADEG"
    AA_tokens = [tokenizer.convert_tokens_to_ids(AA) for AA in list(AAs)]

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
        probabilities = sm(mask_hidden_state)

        # Get a dictionary of AA and probability of it being there at given position
        result = dict(zip(list(AAs), [probabilities[AA].item() for AA in AA_tokens]))
        result["position"] = i

        # Append orderly to mutagenesis
        probabilities_list.append(result)

        if pbar:
            pbar.update()

    return probabilities_list


def plot(
    sequence: str,
    probabilities: List[Dict[str, float]],
    original_id: str,
    start: Optional[int] = None,
    stop: Optional[int] = None,
):
    if not start:
        start = 0
    if not stop:
        stop = len(sequence)

    probabilities_dataframe = (
        DataFrame(probabilities, index=list(sequence[start:stop]))
        .drop("position", axis=1)
        .T
    )

    x_labels = list((f"{i + 1} {AA}" for (i, AA) in enumerate(sequence)))

    values = probabilities_dataframe.values

    filled_in_sequence = numpy.concatenate(
        [
            numpy.full((values.shape[0], start), -1),
            values,
            numpy.full(
                (values.shape[0], len(sequence) - stop),
                -1,
            ),
        ],
        axis=1,
    )

    fig = px.imshow(
        filled_in_sequence,
        labels=dict(x="WT sqeuence", y="AA", color="Probability"),
        color_continuous_scale="blues",
        x=x_labels,
        y=probabilities_dataframe.index.values,
        zmin=0,
        zmax=1,
        width=len(x_labels) * 20,
        title=original_id,
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            tickmode="linear",
        ),
        yaxis=dict(
            tickmode="linear",
        ),
    )

    fig.add_trace(go.Scatter(x=x_labels, y=list(sequence), mode="markers"))

    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)

    return fig

# TODO: Move this once everything works
import re
from itertools import zip_longest
from typing import List, Generator

import torch
from numpy import ndarray

from bio_embeddings.embed.embedder_interface import EmbedderInterface


def embed_batch_berts(
    embedder: EmbedderInterface, batch: List[str]
) -> Generator[ndarray, None, None]:
    """ Embed batch code shared between Bert and Albert """
    seq_lens = [len(seq) for seq in batch]
    # Remove rare amino acids
    batch = [re.sub(r"[UZOB]", "X", sequence) for sequence in batch]
    # transformers needs spaces between the amino acids
    batch = [" ".join(list(seq)) for seq in batch]

    ids = embedder._tokenizer.batch_encode_plus(
        batch, add_special_tokens=True, pad_to_max_length=True
    )

    input_ids = torch.tensor(ids["input_ids"]).to(embedder._device)
    attention_mask = torch.tensor(ids["attention_mask"]).to(embedder._device)

    with torch.no_grad():
        embeddings = embedder._model(input_ids=input_ids, attention_mask=attention_mask)

    embeddings = embeddings[0].cpu().numpy()

    for seq_num, seq_len in zip_longest(range(len(embeddings)), seq_lens):
        # slice off first and last positions (special tokens)
        embedding = embeddings[seq_num][1:-1]
        assert seq_len == embedding.shape[0]
        yield embedding

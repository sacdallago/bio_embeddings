# TODO: Move this once everything works
import re
from typing import List, Generator

import torch
from numpy import ndarray

from bio_embeddings.embed.embedder_interface import EmbedderInterface


def embed_batch_berts(
    embedder: EmbedderInterface, batch: List[str]
) -> Generator[ndarray, None, None]:
    """ Embed batch code shared between Bert and Albert """
    batch    = [ re.sub(r"[UZOB]", "X", sequence) for sequence in batch ]
    seq_lens = [ len(seq)  for seq in batch ]
    batch    = [ ' '.join(list(seq)) for seq in batch ]

    ids = embedder._tokenizer.batch_encode_plus(
        batch, add_special_tokens=True, pad_to_max_length=True
    )
    
    input_ids      = torch.tensor(ids["input_ids"]).to(embedder._device)
    attention_mask = torch.tensor(ids["attention_mask"]).to(embedder._device)
    
    with torch.no_grad():
        embedding = embedder._model(input_ids=input_ids, attention_mask=attention_mask)
        
    embedding = embedding[0].cpu().numpy()
    
    for seq_num in range(len(embedding)):
        seq_len = seq_lens[seq_num]
        assert seq_len == embedding[seq_num].shape[0]+2 # add two to account for special tokens
        seq_emd = embedding[seq_num][1:-1] # slice off first and last positions (special tokens)
        yield seq_emd

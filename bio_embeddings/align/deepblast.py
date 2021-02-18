import math
from typing import List, Tuple, Dict

import numpy
import torch
from deepblast.dataset.alphabet import UniprotTokenizer
from deepblast.dataset.utils import pack_sequences, revstate_f, states2alignment
from deepblast.trainer import LightningAligner
from tqdm import tqdm


def deepblast_align(
    pairings: List[Tuple[str, str]],
    query_by_id: Dict[str, str],
    target_by_id: Dict[str, str],
    model_file: str,
    device: torch.device,
    batch_size: int,
) -> List[Tuple[str, str, str, str]]:
    """Aligns the given pairings using DeepBLAST

    Returns a list of query id, target id, query aligned, target aligned

    The model on its own takes between 740MiB (Titan X, torch 1.5) and 1284MiB (RTX 8000, torch 1.7)

    Note that the batch size has much less of an impact for DeepBLAST than for the embedders
    """
    model = LightningAligner.load_from_checkpoint(model_file).to(device)
    tokenizer = UniprotTokenizer()
    alignments = []
    # Naive batching
    batches = numpy.array_split(pairings, math.ceil(len(pairings) / batch_size))
    for batch in tqdm(batches):
        # noinspection PyArgumentList
        queries = [
            torch.Tensor(tokenizer(query_by_id[query].encode())).long()
            for query, _ in batch
        ]
        # noinspection PyArgumentList
        targets = [
            torch.Tensor(tokenizer(target_by_id[target].encode())).long()
            for _, target in batch
        ]
        seqs, order = pack_sequences(queries, targets)
        gen = model.aligner.traceback(seqs.to(device), order)
        for (decoded, _), (query, target) in zip(gen, batch):
            pred_x, pred_y, pred_states = zip(*decoded)
            pred_alignment = "".join(list(map(revstate_f, pred_states)))
            x_aligned, y_aligned = states2alignment(
                pred_alignment, query_by_id[query], target_by_id[target]
            )
            alignments.append((query, target, x_aligned, y_aligned))
    return alignments

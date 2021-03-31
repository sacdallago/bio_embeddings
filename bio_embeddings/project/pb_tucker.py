from pathlib import Path
from typing import Union

import torch
from torch import nn
from torch import tensor


class PBTucker(nn.Module):
    """Tucker is a contrastive learning model trained to distinguish CATH superfamilies.

    It consumes prottrans_bert_bfd embeddings and reduces the embedding dimensionality from 1024 to 128.
    See https://www.biorxiv.org/content/10.1101/2021.01.21.427551v1

    To use it outside of the pipeline, first instantiate it with
    `pb_tucker = PBTucker.from_file("/path/to/model", device)`,
    then project your reduced bert embedding with
    `pb_tucker.single_pass(torch.tensor(bert_embedding, device=device)).cpu().numpy()`.
    """
    name: str = "pb_tucker"

    def __init__(self):
        super(PBTucker, self).__init__()
        self.tucker = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
        )

    @staticmethod
    def from_file(model_file: Union[str, Path], device: torch.device) -> "PBTucker":
        model = PBTucker()
        model.load_state_dict(torch.load(model_file, map_location=device)["state_dict"])
        model.eval()
        return model.to(device)

    def single_pass(self, x: tensor) -> tensor:
        return self.tucker(x)

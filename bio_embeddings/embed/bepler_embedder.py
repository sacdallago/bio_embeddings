"""
Most of this implementation is taken from
https://github.com/tbepler/protein-sequence-embedding-iclr2019/blob/3bb338bd70e2b7b97c733304d50cfcac9c35cb27/embed_sequences.py

---

Supporting torch > 1.3 was a bit tricky (https://github.com/tbepler/protein-sequence-embedding-iclr2019/issues/21).
Here's what I did:

First, download and unpack the weights to `pretrained_weights`.
```
wget http://bergerlab-downloads.csail.mit.edu/bepler-protein-sequence-embeddings-from-structure-iclr2019/pretrained_models.tar.gz
tar xf pretrained_models.tar.gz
```

Create a torch 1.3 virtualenv (with python 3.7). In this venv, run:

```shell_script
git clone https://github.com/tbepler/protein-sequence-embedding-iclr2019
cd protein-sequence-embedding-iclr2019
pip install Cython
python setup.py install
```

```python
import torch

model = torch.load("pretrained_models/ssa_L1_100d_lstm3x512_lm_i512_mb64_tau0.5_lambda0.1_p0.05_epoch100.sav")
# For some reason torch seems to have missed that in older versions, but requires it in newer ones
state_dict = model.state_dict()
state_dict["scop_predict.gap"] = torch.FloatTensor([-10])
torch.save(state_dict, "pretrained_models/ssa_L1_100d_lstm3x512_lm_i512_mb64_tau0.5_lambda0.1_p0.05_epoch100_updated.state_dict")
print(model)
```

I then switched back to my normal torch 1.5.1 environment and recreated the
printed model params as you can see in the __init__ function.
"""

from typing import Union

import numpy
import torch
from bepler.alphabets import Uniprot21
from bepler.models.embedding import StackedRNN
from bepler.models.multitask import SCOPCM
from bepler.models.sequence import BiLM
from numpy import ndarray
from torch import nn

from bio_embeddings.embed import EmbedderInterface


def _unstack_lstm(lstm, device: torch.device):
    in_size = lstm.input_size
    hidden_dim = lstm.hidden_size
    layers = []
    for i in range(lstm.num_layers):
        layer = nn.LSTM(in_size, hidden_dim, batch_first=True, bidirectional=True)
        layer.to(device)

        attributes = ["weight_ih_l", "weight_hh_l", "bias_ih_l", "bias_hh_l"]
        for attr in attributes:
            dest = attr + "0"
            src = attr + str(i)
            getattr(layer, dest).data[:] = getattr(lstm, src)

            dest = attr + "0_reverse"
            src = attr + str(i) + "_reverse"
            getattr(layer, dest).data[:] = getattr(lstm, src)
        layer.flatten_parameters()
        layers.append(layer)
        in_size = 2 * hidden_dim
    return layers


class BeplerEmbedder(EmbedderInterface):
    """Bepler Embedder

    Bepler, Tristan, and Bonnie Berger. "Learning protein sequence embeddings using information from structure."
    arXiv preprint arXiv:1902.08661 (2019).
    """

    name = "bepler"
    embedding_dimension = 121  # 100 + len(self.alphabet)
    number_of_layers = 1

    # This is derived from ssa_L1_100d_lstm3x512_lm_i512_mb64_tau0.5_lambda0.1_p0.05_epoch100.sav
    # See text at the top of the file
    necessary_files = ["model_file"]

    def __init__(self, device: Union[None, str, torch.device] = None, **kwargs):
        super().__init__(device, **kwargs)
        self.alphabet = Uniprot21()

        # These parameters are part of the model, but we can't load them if we
        # use the state dict
        hidden = 512
        out = 100

        lm = BiLM(
            len(self.alphabet) + 1,
            len(self.alphabet),
            len(self.alphabet),
            hidden * 2,
            2,
        )
        embedding = StackedRNN(
            len(self.alphabet),
            hidden,
            hidden,
            out,
            nlayers=3,
            dropout=0,
            lm=lm,
        )

        self.model = SCOPCM(embedding)
        self.model.load_state_dict(torch.load(self._options["model_file"]))
        self.model = self.model.eval().to(self._device)

        self.lstm_stack = _unstack_lstm(self.model.embedding.rnn, self._device)

    def embed(self, sequence: str) -> ndarray:
        # https://github.com/sacdallago/bio_embeddings/issues/116
        if not sequence:
            return numpy.zeros((0, self.embedding_dimension))
        x = sequence.upper().encode()
        # convert to alphabet index
        x = self.alphabet.encode(x)
        x = torch.from_numpy(x).to(self._device)
        # embed the sequence
        with torch.no_grad():
            x = x.long().unsqueeze(0)
            zs = []
            # noinspection PyUnresolvedReferences
            x_onehot = x.new(x.size(0), x.size(1), 21).float().zero_()
            x_onehot.scatter_(2, x.unsqueeze(2), 1)
            zs.append(x_onehot)
            h = self.model.embedding.embed(x)
            for lstm in self.lstm_stack:
                h, _ = lstm(h)
            h = self.model.embedding.proj(h.squeeze(0)).unsqueeze(0)
            zs.append(h)
            z1 = torch.cat(zs, 2)
            z = z1
            return z.squeeze(0).cpu().numpy()

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        return embedding.mean(axis=0)

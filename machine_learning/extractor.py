import torch
from os import path
from pathlib import Path
from allennlp.commands.elmo import ElmoEmbedder


_model_dir = path.join(Path(path.abspath(__file__)).parent, 'model')
_weights_file_name = 'weights.hdf5'
_options_file_name = 'options.json'


_weight_file = path.join(_model_dir, _weights_file_name)
_options_file = path.join(_model_dir, _options_file_name)
# use GPU if available, otherwise run on CPU

if torch.cuda.is_available():
    print("CUDA available")
    _cuda_device = 0
else:
    print("CUDA NOT available")
    _cuda_device = -1

model = ElmoEmbedder(weight_file=_weight_file, options_file=_options_file, cuda_device=_cuda_device)


def get_seqvec(seq):
    """
        Input:
            seq=amino acid sequence
            model_dir = directory holding weights and parameters of pre-trained ELMo
        Returns:
            Embedding for the amino acid sequence 'seq'
    """

    embedding = model.embed_sentence(list(seq)) # get embedding for sequence

    return embedding

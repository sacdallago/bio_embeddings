from pathlib import Path

import torch
from allennlp.commands.elmo import ElmoEmbedder

model_dir = Path('./model')
model = get_elmo_model(model_dir)  # get pre-trained ELMo


def _get_elmo_model(model_dir):
    """
        Input: Directory holding weights and parameters of per-trained ELMo
        Returns: Instance of ELMo
    """

    weight_file = model_dir / 'weights.hdf5'
    options_file = model_dir / 'options.json'

    # use GPU if available, otherwise run on CPU
    cuda_device = 0 if torch.cuda.is_available() else -1

    return ElmoEmbedder(weight_file=weight_file, options_file=options_file, cuda_device=cuda_device)


def get_seqvec(seq, model_dir):
    """
        Input:
            seq=amino acid sequence
            model_dir = directory holding weights and parameters of pre-trained ELMo
        Returns:
            Embedding for the amino acid sequence 'seq'
    """

    embedding = model.embed_sentence( list(seq) ) # get embedding for sequence

    return embedding


def main():
    # Path to directory holding pre-trained ELMo

    # Test sequence taken from CASP13: 'T1008'
    seq = 'TDELLERLRQLFEELHERGTEIVVEVHINGERDEIRVRNISKEELKKLLERIREKIEREGSSEVEVNVHSGGQTWTFNEK'
    # Takes sequence, returns embedding of shape (3,L,1024) as List-of-Lists (no numpy!)
    embeddings = get_seqvec( seq, model_dir )
    # Sanity check(s)
    print(embeddings.shape)
    print(embeddings)
    assert len(seq) == embeddings.shape[1]
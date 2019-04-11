# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:16:25 2019

@author: Michael
"""

from pathlib import Path

import torch
import torch.nn as nn
from allennlp.commands.elmo import ElmoEmbedder

# Device configuration. Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# return an un-trained model for the prediction of subcellular localization
# in 10 states and a binary classification into membrane-bound vs. soluble
class SUBCELL_FNN(nn.Module):

    def __init__(self):
        super(SUBCELL_FNN, self).__init__()
        # Linear layer, taking embedding dimension 1024 to make predictions:
        self.layer = nn.Sequential(
            nn.Linear(1024, 32),  # in, out
            nn.Dropout(0.25),  # dropout
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.loc_classifier = nn.Linear(32, 10)
        self.mem_classifier = nn.Linear(32, 2)

    def forward(self, x):
        out = self.layer(x)  # map 1024-dimensional ELMo vector to 32-dims
        # based on 32 dims, predict localization and membrane-bound
        Yhat_loc = self.loc_classifier(out)
        Yhat_mem = self.mem_classifier(out)

        return Yhat_loc, Yhat_mem

    # Convolutional neural network for prediction of Sec.Struct. in 3- & 8-states and disorder


class SECSTRUCT_CNN(nn.Module):
    def __init__(self):
        super(SECSTRUCT_CNN, self).__init__()

        self.elmo_feature_extractor = nn.Sequential(
            nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.dssp3_classifier = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=(7, 1), padding=(3, 0))
        )
        self.dssp8_classifier = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=(7, 1), padding=(3, 0))
        )
        self.diso_classifier = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=(7, 1), padding=(3, 0))
        )

    def forward(self, x):
        x = self.elmo_feature_extractor(x)  # compress ELMo features to 32-dims

        d3_Yhat = self.dssp3_classifier(x)
        d8_Yhat = self.dssp8_classifier(x)
        diso_Yhat = self.diso_classifier(x)

        return d3_Yhat, d8_Yhat, diso_Yhat


def get_loc(embedding, model_dir):
    # 0) Class : Label mapping
    loc_labels = {0: 'Cell-Membrane',
                  1: 'Cytoplasm',
                  2: 'Endoplasmic reticulum',
                  3: 'Golgi-Apparatus',
                  4: 'Lysosome/Vacuole',
                  5: 'Mitochondrion',
                  6: 'Nucleus',
                  7: 'Peroxisome',
                  8: 'Plastid',
                  9: 'Extra-cellular'}

    mem_labels = {0: 'Soluble', 1: 'Membrane-bound'}

    # 1) Read in pre-trained model (again: we can save time by holding it in main memory)
    model = SUBCELL_FNN().to(device)  # create un-trained (raw) model
    state = torch.load(model_dir / 'subcell_checkpoint.pt')  # load pre-trained weights
    model.load_state_dict(state['state_dict'])  # load pre-trained weights into raw model
    model.eval()  # ensure that model is in evaluation mode (important for batchnorm and dropout)

    # 2) Get predictions
    embedding = torch.tensor(embedding).to(device).sum(dim=0).mean(dim=0, keepdim=True)
    yhat_loc, yhat_mem = model(embedding)

    # 3) Map predictions to labels
    pred_loc = loc_labels[torch.max(yhat_loc, dim=1)[1].item()]  # get index of output node with max. activation,
    pred_mem = mem_labels[torch.max(yhat_mem, dim=1)[1].item()]  # this corresponds to the predicted class
    return pred_loc, pred_mem


def get_secstruct(embedding, model_dir):
    def _class2label(label_dict, yhat):
        # get index of output node with max. activation (=predicted class)
        class_indices = torch.max(yhat, dim=1)[1].squeeze()
        yhat = [label_dict[class_idx.item()] for class_idx in class_indices]
        return ''.join(yhat)

    # 0) Class : Label mapping
    dssp8_labels = {0: 'G', 1: 'H', 2: 'I', 3: 'B', 4: 'E', 5: 'S', 6: 'T', 7: 'C'}  # GHIBESTC
    dssp3_labels = {0: 'H', 1: 'E', 2: 'C'}  # Helix, Sheet, Other
    disor_labels = {0: '-', 1: 'X'}  # disorder = unresolved = 'X'

    # 1) Read in pre-trained model (again: we can save time by holding it in main memory)
    model = SECSTRUCT_CNN().to(device)  # create un-trained (raw) model
    state = torch.load(model_dir / 'secstruct_checkpoint.pt')  # load pre-trained weights
    model.load_state_dict(state['state_dict'])  # load pre-trained weights into raw model
    model.eval()  # ensure that model is in evaluation mode (important for batchnorm and dropout)

    # 2) Get predictions
    # Sum over 3 ELMo layers and add singleton dimension to fit CNN requirements
    embedding = torch.tensor(embedding).to(device).sum(dim=0, keepdim=True).permute(0, 2, 1).unsqueeze(dim=-1)
    yhat_dssp3, yhat_dssp8, yhat_disor = model(embedding)

    # 3) Map predictions to labels
    pred_dssp3 = _class2label(dssp3_labels, yhat_dssp3)
    pred_dssp8 = _class2label(dssp8_labels, yhat_dssp8)
    pred_disor = _class2label(disor_labels, yhat_disor)
    return pred_dssp3, pred_dssp8, pred_disor


def get_elmo_model(model_dir):
    '''
        Input: Directory holding weights and parameters of per-trained ELMo
        Returns: Instance of ELMo
    '''
    weight_file = model_dir / 'weights.hdf5'
    options_file = model_dir / 'options.json'
    # use GPU if available, otherwise run on CPU
    cuda_device = 0 if torch.cuda.is_available() else -1
    return ElmoEmbedder(weight_file=weight_file, options_file=options_file, cuda_device=cuda_device)


def get_seqvec(seq, model_dir):
    '''
        Input:
            seq=amino acid sequence
            model_dir = directory holding weights and parameters of pre-trained ELMo
        Returns:
            Embedding for the amino acid sequence 'seq'
    '''
    model = get_elmo_model(model_dir)  # get pre-trained ELMo
    embedding = model.embed_sentence(list(seq))  # get embedding for sequence

    return embedding


def main():
    # Path to directory holding pre-trained ELMo (Uniref50_v2) and pre-trained task-specific models
    model_dir = Path.cwd() / 'models'
    elmo_dir = model_dir / 'uniref50_v2'

    # Test sequence taken from CASP13: 'T1008'
    seq = 'TDELLERLRQLFEELHERGTEIVVEVHINGERDEIRVRNISKEELKKLLERIREKIEREGSSEVEVNVHSGGQTWTFNEK'

    # Takes sequence, returns embedding of shape (3,L,1024) as List-of-Lists (no numpy!)
    embeddings = get_seqvec(seq, elmo_dir)

    # Sanity check(s)
    assert len(seq) == embeddings.shape[1]

    # Retrieve task-specific predictions based on ELMo embeddings
    pred_loc, pred_mem = get_loc(embeddings, model_dir)
    print('Predicted to be in {} and being {}'.format(pred_loc, pred_mem))

    dssp3, dssp8, disor = get_secstruct(embeddings, model_dir)
    print('Predicted DSSP3:\n{}'.format(dssp3))
    print('Predicted DSSP8:\n{}'.format(dssp8))
    print('Predicted disorder:\n{}'.format(disor))
    assert len(seq) == len(dssp3) and len(seq) == len(dssp8) and len(seq) == len(disor)


if __name__ == '__main__':
    main()

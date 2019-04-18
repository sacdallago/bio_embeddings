import torch
from os import path
from pathlib import Path
from app.machine_learning.models import SUBCELL_FNN, SECSTRUCT_CNN

_model_dir = path.join(Path(path.abspath(__file__)).parent, 'model')
_subcellular_location_checkpoint = 'subcell_checkpoint.pt'
_secondary_structure_checkpoint = 'secstruct_checkpoint.pt'


def get_subcellular_location(embedding):
    # new device TODO: might not need
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    model = SUBCELL_FNN().to(_device)  # create un-trained (raw) model

    if torch.cuda.is_available():
        state = torch.load(path.join(_model_dir, _subcellular_location_checkpoint))  # load pre-trained weights
    else:
        state = torch.load(path.join(_model_dir, _subcellular_location_checkpoint), map_location='cpu')  # load pre-trained weights

    model.load_state_dict(state['state_dict'])  # load pre-trained weights into raw model
    model.eval()  # ensure that model is in evaluation mode (important for batchnorm and dropout)

    # 2) Get predictions
    embedding = torch.tensor(embedding).to(_device).sum(dim=0).mean(dim=0, keepdim=True)
    yhat_loc, yhat_mem = model(embedding)

    # 3) Map predictions to labels
    pred_loc = loc_labels[torch.max(yhat_loc, dim=1)[1].item()]  # get index of output node with max. activation,
    pred_mem = mem_labels[torch.max(yhat_mem, dim=1)[1].item()]  # this corresponds to the predicted class

    return pred_loc, pred_mem


def get_secondary_structure(embedding):
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 0) Class : Label mapping
    dssp8_labels = {0: 'G', 1: 'H', 2: 'I', 3: 'B', 4: 'E', 5: 'S', 6: 'T', 7: 'C'}  # GHIBESTC
    dssp3_labels = {0: 'H', 1: 'E', 2: 'C'}  # Helix, Sheet, Other
    disor_labels = {0: '-', 1: 'X'}  # disorder = unresolved = 'X'

    # 1) Read in pre-trained model (again: we can save time by holding it in main memory)
    model = SECSTRUCT_CNN().to(_device)  # create un-trained (raw) model

    if torch.cuda.is_available():
        state = torch.load(path.join(_model_dir, _secondary_structure_checkpoint))  # load pre-trained weights
    else:
        state = torch.load(path.join(_model_dir, _secondary_structure_checkpoint), map_location='cpu')  # load pre-trained weights
    model.load_state_dict(state['state_dict'])  # load pre-trained weights into raw model
    model.eval()  # ensure that model is in evaluation mode (important for batchnorm and dropout)

    # 2) Get predictions
    # Sum over 3 ELMo layers and add singleton dimension to fit CNN requirements
    embedding = torch.tensor(embedding).to(_device).sum(dim=0, keepdim=True).permute(0, 2, 1).unsqueeze(dim=-1)
    yhat_dssp3, yhat_dssp8, yhat_disor = model(embedding)

    # 3) Map predictions to labels
    pred_dssp3 = _class2label(dssp3_labels, yhat_dssp3)
    pred_dssp8 = _class2label(dssp8_labels, yhat_dssp8)
    pred_disor = _class2label(disor_labels, yhat_disor)

    return pred_dssp3, pred_dssp8, pred_disor


def _class2label(label_dict, yhat):
    # get index of output node with max. activation (=predicted class)
    class_indices = torch.max(yhat, dim=1)[1].squeeze()
    yhat = [label_dict[class_idx.item()] for class_idx in class_indices]
    return ''.join(yhat)

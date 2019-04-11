import torch
from os import path
from pathlib import Path
from app.machine_learning.models import SECSTRUCT_CNN, SUBCELL_FNN
from allennlp.commands.elmo import ElmoEmbedder


_model_dir = path.join(Path(path.abspath(__file__)).parent, 'model')
_weights_file_name = 'weights.hdf5'
_options_file_name = 'options.json'
_subcellular_location_checkpoint = 'subcell_checkpoint.pt'
_secondary_structure_checkpoint = 'secstruct_checkpoint.pt'


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
    state = torch.load(path.join(_model_dir, _subcellular_location_checkpoint))  # load pre-trained weights
    model.load_state_dict(state['state_dict'])  # load pre-trained weights into raw model
    model.eval()  # ensure that model is in evaluation mode (important for batchnorm and dropout)

    # 2) Get predictions
    embedding = torch.tensor(embedding).to(_device).sum(dim=0).mean(dim=0, keepdim=True)
    yhat_loc, yhat_mem = model(embedding)

    # 3) Map predictions to labels
    pred_loc = loc_labels[torch.max(yhat_loc, dim=1)[1].item()]  # get index of output node with max. activation,
    pred_mem = mem_labels[torch.max(yhat_mem, dim=1)[1].item()]  # this corresponds to the predicted class

    return pred_loc, pred_mem


def _class2label(label_dict, yhat):
    # get index of output node with max. activation (=predicted class)
    class_indices = torch.max(yhat, dim=1)[1].squeeze()
    yhat = [label_dict[class_idx.item()] for class_idx in class_indices]
    return ''.join(yhat)


def get_secondary_structure(embedding):
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 0) Class : Label mapping
    dssp8_labels = {0: 'G', 1: 'H', 2: 'I', 3: 'B', 4: 'E', 5: 'S', 6: 'T', 7: 'C'}  # GHIBESTC
    dssp3_labels = {0: 'H', 1: 'E', 2: 'C'}  # Helix, Sheet, Other
    disor_labels = {0: '-', 1: 'X'}  # disorder = unresolved = 'X'

    # 1) Read in pre-trained model (again: we can save time by holding it in main memory)
    model = SECSTRUCT_CNN().to(_device)  # create un-trained (raw) model
    state = torch.load(path.join(_model_dir, _secondary_structure_checkpoint))  # load pre-trained weights
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

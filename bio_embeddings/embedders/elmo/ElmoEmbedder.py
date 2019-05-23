import torch
from bio_embeddings.embedders.EmbedderInterface import EmbedderInterface
from bio_embeddings.features import Location, Membrane, Disorder, SecondaryStructure
from bio_embeddings.utilities import Logger, get_defaults
from allennlp.commands.elmo import ElmoEmbedder as _ElmoEmbedder


class ElmoEmbedder(EmbedderInterface):

    def __init__(self, **kwargs):
        """
        Initialize Elmo embedder. Can define non-positional arguments for paths of files and version of ELMO.

        If version is supplied, paths will be ignored and model will be downloaded from remote location.

        If one of the files is not supplied, all the files will be downloaded.

        :param weights_file: path of weights file
        :param options_file: path of options file
        :param secondary_structure_checkpoint_file: path of secondary structure checkpoint file
        :param subcellular_location_checkpoint: path of the subcellular location checkpoint file
        :param version: Integer. Available versions: 1, 2


        """
        super().__init__()

        self._options = kwargs

        self._weights_file = self._options.get('weights_file')
        self._options_file = self._options.get('options_file')
        self._secondary_structure_checkpoint_file = self._options.get('secondary_structure_checkpoint_file')
        self._subcellular_location_checkpoint = self._options.get('subcellular_location_checkpoint')

        # use GPU if available, otherwise run on CPU
        # !important: GPU visibility can easily be hidden using this env variable: CUDA_VISIBLE_DEVICES=""
        # This is especially useful if using an old CUDA device which is not supported by pytorch!

        if torch.cuda.is_available():
            Logger.log("CUDA available")
            _cuda_device = 0
        else:
            Logger.log("CUDA NOT available")
            _cuda_device = -1

        version = self._options.get('version')

        if version is not None and version in [1, 2]:
            if version == 1:
                self._weight_file, self._options_file, self._subcellular_location_checkpoint, self._secondary_structure_checkpoint_file = get_defaults(
                    'elmov1')
            elif version == 2:
                self._weight_file, self._options_file, self._subcellular_location_checkpoint, self._secondary_structure_checkpoint_file = get_defaults(
                    'elmov2')
        elif self._weights_file is None or \
              self._options_file is None or \
              self._subcellular_location_checkpoint is None or \
              self._secondary_structure_checkpoint_file is None:

            self._weight_file, self._options_file, self._subcellular_location_checkpoint, self._secondary_structure_checkpoint_file = get_defaults('elmov1')

        self._model = _ElmoEmbedder(weight_file=self._weight_file,
                                    options_file=self._options_file,
                                    cuda_device=_cuda_device)

        pass


    def embed(self, sequence):

        # TODO: Test that sequence is a valid sequence

        self._sequence = sequence
        self._embedding = self._model.embed_sentence(list(self._sequence))  # get embedding for sequence

        return self._embedding.tolist()

    def get_features(self):
        # Will raise exception if no embedding
        self.get_embedding()


        pass

    def _get_subcellular_location(self):
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
            state = torch.load(path.join(_model_dir, _subcellular_location_checkpoint),
                               map_location='cpu')  # load pre-trained weights

        model.load_state_dict(state['state_dict'])  # load pre-trained weights into raw model
        model.eval()  # ensure that model is in evaluation mode (important for batchnorm and dropout)

        # 2) Get predictions
        embedding = torch.tensor(self._embedding).to(_device).sum(dim=0).mean(dim=0, keepdim=True)
        yhat_loc, yhat_mem = model(embedding)

        # 3) Map predictions to labels
        pred_loc = loc_labels[torch.max(yhat_loc, dim=1)[1].item()]  # get index of output node with max. activation,
        pred_mem = mem_labels[torch.max(yhat_mem, dim=1)[1].item()]  # this corresponds to the predicted class

        return pred_loc, pred_mem

    def _get_secondary_structure(self):
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
            state = torch.load(path.join(_model_dir, _secondary_structure_checkpoint),
                               map_location='cpu')  # load pre-trained weights
        model.load_state_dict(state['state_dict'])  # load pre-trained weights into raw model
        model.eval()  # ensure that model is in evaluation mode (important for batchnorm and dropout)

        # 2) Get predictions
        # Sum over 3 ELMo layers and add singleton dimension to fit CNN requirements
        embedding = torch.tensor(self._embedding).to(_device).sum(dim=0, keepdim=True).permute(0, 2, 1).unsqueeze(dim=-1)
        yhat_dssp3, yhat_dssp8, yhat_disor = model(embedding)

        # 3) Map predictions to labels
        pred_dssp3 = self._class2label(dssp3_labels, yhat_dssp3)
        pred_dssp8 = self._class2label(dssp8_labels, yhat_dssp8)
        pred_disor = self._class2label(disor_labels, yhat_disor)

        return pred_dssp3, pred_dssp8, pred_disor

    def _class2label(label_dict, yhat):
        # get index of output node with max. activation (=predicted class)
        class_indices = torch.max(yhat, dim=1)[1].squeeze()
        yhat = [label_dict[class_idx.item()] for class_idx in class_indices]

        return ''.join(yhat)

import torch
from bio_embeddings.embedders.EmbedderInterface import EmbedderInterface, NoEmbeddingException
from bio_embeddings.features import Location, Membrane, Disorder, SecondaryStructure, FeaturesCollection
from bio_embeddings.embedders.elmo.feature_inference_models import SUBCELL_FNN, SECSTRUCT_CNN
from bio_embeddings.utilities import Logger, get_defaults
from allennlp.commands.elmo import ElmoEmbedder as _ElmoEmbedder

# Label mappings
_loc_labels = {0: 'Cell-Membrane',
               1: 'Cytoplasm',
               2: 'Endoplasmic reticulum',
               3: 'Golgi-Apparatus',
               4: 'Lysosome/Vacuole',
               5: 'Mitochondrion',
               6: 'Nucleus',
               7: 'Peroxisome',
               8: 'Plastid',
               9: 'Extra-cellular'}

_mem_labels = {0: False, 1: True}

_dssp8_labels = {0: 'G', 1: 'H', 2: 'I', 3: 'B', 4: 'E', 5: 'S', 6: 'T', 7: 'C'}  # GHIBESTC
_dssp3_labels = {0: 'H', 1: 'E', 2: 'C'}  # Helix, Sheet, Other, HEC
_disor_labels = {0: '-', 1: 'X'}  # disorder = unresolved = 'X'


class ElmoEmbedder(EmbedderInterface):

    def __init__(self, **kwargs):
        """
        Initialize Elmo embedder. Can define non-positional arguments for paths of files and version of ELMO.

        If version is supplied, paths will be ignored and model will be downloaded from remote location.

        If one of the files is not supplied, all the files will be downloaded.

        :param weights_file: path of weights file
        :param options_file: path of options file
        :param secondary_structure_checkpoint_file: path of secondary structure checkpoint file
        :param subcellular_location_checkpoint_file: path of the subcellular location checkpoint file
        :param version: Integer. Available versions: 1, 2


        """
        super().__init__()

        self._options = kwargs

        # Get file locations from kwargs
        self._weights_file = self._options.get('weights_file')
        self._options_file = self._options.get('options_file')
        self._secondary_structure_checkpoint_file = self._options.get('secondary_structure_checkpoint_file')
        self._subcellular_location_checkpoint_file = self._options.get('subcellular_location_checkpoint_file')

        # Get preferred version, if defined
        version = self._options.get('version')

        # If version defined: fetch online
        if version is not None and version in [1, 2]:
            if version == 1:
                self._temp_weights_file, self._temp_options_file, self._temp_subcellular_location_checkpoint_file, self._temp_secondary_structure_checkpoint_file = get_defaults(
                    'elmov1')
            elif version == 2:
                self._temp_weights_file, self._temp_options_file, self._temp_subcellular_location_checkpoint_file, self._temp_secondary_structure_checkpoint_file = get_defaults(
                    'elmov2')

            self._weights_file, self._options_file, self._subcellular_location_checkpoint_file, self._secondary_structure_checkpoint_file = self._temp_weights_file.name, self._temp_options_file.name, self._temp_subcellular_location_checkpoint_file.name, self._temp_secondary_structure_checkpoint_file.name

        # If any file is not defined: fetch all files online
        elif self._weights_file is None or \
              self._options_file is None or \
              self._subcellular_location_checkpoint_file is None or \
              self._secondary_structure_checkpoint_file is None:

            self._temp_weights_file, self._temp_options_file, self._temp_subcellular_location_checkpoint_file, self._temp_secondary_structure_checkpoint_file = get_defaults('elmov1')

            self._weights_file, self._options_file, self._subcellular_location_checkpoint_file, self._secondary_structure_checkpoint_file = self._temp_weights_file.name, self._temp_options_file.name, self._temp_subcellular_location_checkpoint_file.name, self._temp_secondary_structure_checkpoint_file.name
            pass

        # use GPU if available, otherwise run on CPU
        # !important: GPU visibility can easily be hidden using this env variable: CUDA_VISIBLE_DEVICES=""
        # This is especially useful if using an old CUDA device which is not supported by pytorch!

        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Read in pre-trained model
        # Create un-trained (raw) model
        self._subcellular_location_model = SUBCELL_FNN().to(self._device)
        self._secondary_structure_model = SECSTRUCT_CNN().to(self._device)

        if torch.cuda.is_available():
            Logger.log("CUDA available")

            # load pre-trained weights for feature machines
            subcellular_state = torch.load(self._subcellular_location_checkpoint_file)
            secondary_structure_state = torch.load(self._secondary_structure_checkpoint_file)

            # Set CUDA device for ELMO machine
            _cuda_device = 0
        else:
            Logger.log("CUDA NOT available")

            # load pre-trained weights for feature machines
            subcellular_state = torch.load(self._subcellular_location_checkpoint_file, map_location='cpu')
            secondary_structure_state = torch.load(self._secondary_structure_checkpoint_file, map_location='cpu')

            # Set CUDA device for ELMO machine
            _cuda_device = -1
            pass

        # load pre-trained weights into raw model
        self._subcellular_location_model.load_state_dict(subcellular_state['state_dict'])
        self._secondary_structure_model.load_state_dict(secondary_structure_state['state_dict'])

        # ensure that model is in evaluation mode (important for batchnorm and dropout)
        self._subcellular_location_model.eval()
        self._secondary_structure_model.eval()

        # Load ELMO model
        self._elmo_model = _ElmoEmbedder(weight_file=self._weights_file,
                                         options_file=self._options_file,
                                         cuda_device=_cuda_device)

        pass

    def embed(self, sequence):

        # TODO: Test that sequence is a valid sequence

        self._sequence = sequence
        self._embedding = self._elmo_model.embed_sentence(list(self._sequence))  # get embedding for sequence

        return self._embedding.tolist()

    def get_features(self, embedding=None):
        # Allows to use get_features as a partly static method
        if embedding is not None:
            self._embedding = embedding

        # Will raise exception if no embedding
        self.get_embedding()

        secondary_structure, disorder = self._get_secondary_structure()
        location, membrane = self._get_subcellular_location()

        features = FeaturesCollection()
        features.disorder = disorder
        features.membrane = membrane
        features.location = location
        features.secondaryStructure = secondary_structure

        return features

    def _get_subcellular_location(self):
        embedding = torch.tensor(self._embedding).to(self._device).sum(dim=0).mean(dim=0, keepdim=True)
        yhat_loc, yhat_mem = self._subcellular_location_model(embedding)

        pred_loc = _loc_labels[torch.max(yhat_loc, dim=1)[1].item()]  # get index of output node with max. activation,
        pred_mem = _mem_labels[torch.max(yhat_mem, dim=1)[1].item()]  # this corresponds to the predicted class

        result_loc = Location()
        result_loc.set_location(pred_loc)

        result_membrane = Membrane()
        result_membrane.set_membrane(pred_mem)

        return result_loc, result_membrane

    def _get_secondary_structure(self):
        embedding = torch.tensor(self._embedding).to(self._device).sum(dim=0, keepdim=True).permute(0, 2, 1).unsqueeze(dim=-1)
        yhat_dssp3, yhat_dssp8, yhat_disor = self._secondary_structure_model(embedding)

        pred_dssp3 = self._class2label(_dssp3_labels, yhat_dssp3)
        pred_dssp8 = self._class2label(_dssp8_labels, yhat_dssp8)
        pred_disor = self._class2label(_disor_labels, yhat_disor)

        result_secondary_structure = SecondaryStructure()
        result_secondary_structure.set_DSSP3(pred_dssp3)
        result_secondary_structure.set_DSSP8(pred_dssp8)

        result_disorder = Disorder()
        result_disorder.set_disorder(pred_disor)

        return result_secondary_structure, result_disorder

    @staticmethod
    def _class2label(label_dict, yhat):
        # get index of output node with max. activation (=predicted class)
        class_indices = torch.max(yhat, dim=1)[1].squeeze()
        yhat = [label_dict[class_idx.item()] for class_idx in class_indices]

        return ''.join(yhat)

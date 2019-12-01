import torch
from bio_embeddings.embedders.EmbedderInterface import EmbedderInterface
from bio_embeddings.utilities import Logger, get_model_parameters, CannotInferModelVersionException
from allennlp.commands.elmo import ElmoEmbedder as _ElmoEmbedder


class ElmoEmbedder(EmbedderInterface):

    def __init__(self, **kwargs):
        """
        Initialize Elmo embedder. Can define non-positional arguments for paths of files and version of ELMO.

        If no version is supplied, v1 will be assumed. If version is set to 1 but vocabulary file is supplied,
        will throw error.

        If one of the files is not supplied, all the files will be downloaded.

        :param weights_file: path of weights file
        :param options_file: path of options file
        :param vocabulary_file: path of vocabulary file. Only needed if seqveq v2.

        :param version: Integer. Available versions: 1, 2
        """
        super().__init__()

        self._options = kwargs

        # Get file locations from kwargs
        self._vocabulary_file = self._options.get('vocabulary_file')
        self._weights_file = self._options.get('weights_file')
        self._options_file = self._options.get('options_file')

        # Get preferred version, if defined
        self._version = self._options.get('version')

        if self._version is None and self._vocabulary_file is not None:
            self._version = 2
        elif self._version is None and self._vocabulary_file is None:
            self._version = 1
        elif self._version is not None and self._version not in [1, 2]:
            raise CannotInferModelVersionException
        elif self._version == 1 and self._vocabulary_file is not None:
            raise CannotInferModelVersionException

        # If any file is not defined: fetch all files online
        if self._version == 1:
            necessary_files = ['weights_file', 'options_file']

            if not set(necessary_files) <= set(self._options.keys()):
                self._temp_weights_file, self._temp_options_file = get_model_parameters('seqvecv1')

                self._weights_file, self._options_file = self._temp_weights_file.name, self._temp_options_file.name
        elif self._version == 1:
            necessary_files = ['vocabulary_file', 'weights_file', 'options_file']

            if not set(necessary_files) <= set(self._options.keys()):
                self._temp_weights_file, self._temp_options_file, self._temp_vocabulary_file = get_model_parameters('elmov2')

                self._weights_file, self._options_file, self._vocabulary_file, = self._temp_weights_file.name, self._temp_options_file.name, self._temp_vocabulary_file.name

        if torch.cuda.is_available():
            Logger.log("CUDA available")

            # Set CUDA device for ELMO machine
            _cuda_device = 0
        else:
            Logger.log("CUDA NOT available")

            # Set CUDA device for ELMO machine
            _cuda_device = -1
            pass

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

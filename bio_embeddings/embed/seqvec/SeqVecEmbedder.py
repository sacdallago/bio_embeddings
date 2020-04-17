import torch
from bio_embeddings.embed.EmbedderInterface import EmbedderInterface
from bio_embeddings.utilities import Logger
from allennlp.commands.elmo import ElmoEmbedder as _ElmoEmbedder


class SeqVecEmbedder(EmbedderInterface):

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
        self._version = self._options.get('seqvec_version')

        if torch.cuda.is_available():
            Logger.log("CUDA available")

            # Set CUDA device for ELMO machine
            _cuda_device = 0
        else:
            Logger.log("CUDA NOT available")

            # Set CUDA device for ELMO machine
            _cuda_device = -1
            pass

        # Set AA lower bound
        self._max_amino_acids = self._options.get('max_amino_acids', 15000)

        self._elmo_model = _ElmoEmbedder(weight_file=self._weights_file,
                                         options_file=self._options_file,
                                         cuda_device=_cuda_device)

        pass

    def embed(self, sequence):
        embedding = self._elmo_model.embed_sentence(list(sequence))  # get embedding for sequence
        return embedding.tolist()

    def embed_many(self, sequences):
        tokenized_sequences = [list(s) for s in sequences]
        candidates = list()
        result = list()
        aa_count = 0

        while tokenized_sequences:
            if aa_count < self._max_amino_acids:
                current = tokenized_sequences.pop()
                aa_count += len(current)
                candidates.append(current)
            else:
                result.extend(list(self._elmo_model.embed_sentences(candidates)))

                # Reset
                aa_count = 0
                candidates = list()

        if candidates:
            result.extend(list(self._elmo_model.embed_sentences(candidates)))

        return result

    @staticmethod
    def reduce_per_protein(embedding):
        return embedding.sum(0).mean(0)

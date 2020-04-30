import torch
from bio_embeddings.embed.EmbedderInterface import EmbedderInterface
from bio_embeddings.utilities import Logger
from allennlp.commands.elmo import ElmoEmbedder as _ElmoEmbedder


class SeqVecEmbedder(EmbedderInterface):

    def __init__(self, **kwargs):
        """
        Initialize Elmo embedder. Can define non-positional arguments for paths of files and other settings.

        :param weights_file: path of weights file
        :param options_file: path of options file
        :param use_cpu: overwrite autodiscovery and force CPU use
        :param max_amino_acids: max # of amino acids to include in embed_many batches. Default: 15k AA

        """
        super().__init__()

        self._options = kwargs

        # Get file locations from kwargs
        self._weights_file = self._options.get('weights_file')
        self._options_file = self._options.get('options_file')
        self._use_cpu = self._options.get('use_cpu', False)

        if torch.cuda.is_available() and not self._use_cpu:
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
                current = tokenized_sequences.pop(0)
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

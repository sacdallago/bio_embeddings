import re
import torch
import numpy as np
from pathlib import Path
from transformers import AlbertModel, AlbertTokenizer
from bio_embeddings.embed.EmbedderInterface import EmbedderInterface
from bio_embeddings.utilities import Logger, SequenceTooLongException, SequenceEmbeddingLengthMismatchException


class AlbertEmbedder(EmbedderInterface):

    def __init__(self, **kwargs):
        """
        Initialize Albert embedder.

        :param model_directory:
        :param ignore_long_proteins: True will ignore proteins longer than 510. False: will throw exception if embedding sequence with length > 510. Default: True
        :param use_cpu: overwrite autodiscovery and force CPU use
        """
        super().__init__()

        self._options = kwargs

        # Get file locations from kwargs
        self._model_directory = self._options.get('model_directory')
        self._ignore_long_proteins = self._options.get('ignore_long_proteins', True)
        self._use_cpu = self._options.get('use_cpu', False)

        # utils
        self._device = torch.device('cuda:0' if torch.cuda.is_available() and not self._use_cpu else 'cpu')
        self._max_sequence_length = 510

        # make model
        self._albert_model = AlbertModel.from_pretrained(self._model_directory)
        self._albert_model = self._albert_model.eval()
        self._tokenizer = AlbertTokenizer(str(Path(self._model_directory) / 'albert_vocab_model.model'), do_lower_case=False)

        pass

    def embed(self, sequence):
        sequence_length = len(sequence)
        if sequence_length > self._max_sequence_length:
            if not self._ignore_long_proteins:
                raise SequenceTooLongException()
            else:
                Logger.log('''Trying to embed a sequence of length {}, but maximal length allowed is {}.
                            The embedding for this sequence will be zeroes!'''.format(
                    sequence_length, self._max_sequence_length
                ))
                return np.zeros((sequence_length, 4096))

        sequence = re.sub(r"[U|Z|O|B]", "X", sequence)

        # Tokenize sequence with spaces
        sequence = ' '.join(list(sequence))

        # tokenize sequence
        tokenized_sequence = torch.tensor([self._tokenizer.encode(sequence, add_special_tokens=True)]).to(self._device)

        with torch.no_grad():
            # drop batch dimension
            embedding = self._albert_model(tokenized_sequence)[0].squeeze()
            # remove special tokens added to start/end
            embedding = embedding[1:sequence_length + 1]

        if not sequence_length == embedding.shape[0]:
            raise SequenceEmbeddingLengthMismatchException()

        return embedding.cpu().detach().numpy().squeeze()

    def embed_many(self, sequences):

        return [self.embed(sequence) for sequence in sequences]

    @staticmethod
    def reduce_per_protein(embedding):
        return embedding.mean(axis=0)

from tempfile import TemporaryDirectory
from typing import Type

import numpy
from numpy import ndarray

from bio_embeddings.embed import EmbedderInterface
from bio_embeddings.embed.embedder_interfaces import EmbedderInterfaceSubclass
from bio_embeddings.embed.pipeline import embed_and_write_batched
from bio_embeddings.utilities.filemanagers import FileSystemFileManager

# noinspection PyProtectedMember
from bio_embeddings.utilities.pipeline import _process_fasta_file


class FakeEmbedder(EmbedderInterface):
    embedding_dimension = 1024
    number_of_layers = 1

    @classmethod
    def with_download(
        cls: Type[EmbedderInterfaceSubclass], **kwargs
    ) -> EmbedderInterfaceSubclass:
        raise NotImplemented

    def embed(self, sequence: str) -> ndarray:
        return numpy.asarray([])

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        return embedding


def test_simple_remapping():
    """ https://github.com/sacdallago/bio_embeddings/issues/50 """
    with TemporaryDirectory() as prefix:
        global_parameters = {
            "sequences_file": "test-data/seqwence-protein.fasta",
            "prefix": prefix,
            "simple_remapping": True,
        }
        global_parameters = _process_fasta_file(**global_parameters)
        embed_and_write_batched(
            FakeEmbedder(),
            FileSystemFileManager(),
            {
                **global_parameters,
                "max_amino_acids": 10000,
                "discard_per_amino_acid_embeddings": False,
            },
        )

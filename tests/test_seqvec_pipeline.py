from tempfile import TemporaryDirectory
from typing import List
from unittest import mock

import numpy
from numpy import ndarray

from bio_embeddings.embed.pipeline import seqvec


class MockElmoEmbedder:
    # noinspection PyUnusedLocal
    def __init__(self, options_file: str, weight_file: str, cuda_device: int = -1):
        self.embeddedings = numpy.load("test-data/embeddings.npz")

    def embed_batch(self, many: List[str]) -> List[ndarray]:
        return [self.embed_sentence(i) for i in many]

    def embed_sentence(self, _sentence: str) -> ndarray:
        return list(self.embeddedings)[0]  # TODO


def test_seqvec():
    with mock.patch(
        "bio_embeddings.embed.seqvec_embedder.ElmoEmbedder", MockElmoEmbedder
    ), TemporaryDirectory() as tmp_dir:
        out = seqvec(
            weights_file="mocked.txt",
            options_file="mocked.txt",
            seqvec_version=1,
            prefix=tmp_dir,
            remapped_sequences_file="test-data/remapped_sequences_file.fasta",
            mapping_file="test-data/mapping_file.csv",
        )
        # TODO: Check output
        print(out)

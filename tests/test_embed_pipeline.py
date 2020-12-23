from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
from unittest import mock
from unittest.mock import MagicMock

import numpy
import pytest
from numpy import ndarray

from bio_embeddings.embed import name_to_embedder
from bio_embeddings.embed.pipeline import run, DEFAULT_MAX_AMINO_ACIDS, ALL_PROTOCOLS
from bio_embeddings.utilities import InvalidParameterError


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
        Path(tmp_dir).joinpath("seqvec_stage").mkdir()
        out = run(
            protocol="seqvec",
            stage_name="seqvec_stage",
            weights_file="mocked.txt",
            options_file="mocked.txt",
            seqvec_version=1,
            prefix=tmp_dir,
            remapped_sequences_file="test-data/remapped_sequences_file.fasta",
            mapping_file="test-data/mapping_file.csv",
        )
        # TODO: Check output
        print(out)


def test_missing_extras():
    """https://github.com/sacdallago/bio_embeddings/issues/105"""
    # Make sure those are in sync
    assert set(name_to_embedder) == set(DEFAULT_MAX_AMINO_ACIDS)
    assert set(name_to_embedder) == set(ALL_PROTOCOLS)
    with mock.patch("bio_embeddings.embed.pipeline.name_to_embedder", {}):
        with pytest.raises(
            InvalidParameterError,
            match="The extra for the protocol seqvec is missing. "
            "See https://docs.bioembeddings.com/#installation on how to install all extras",
        ):
            run(
                protocol="seqvec",
                prefix=MagicMock(),
                stage_name=MagicMock(),
                remapped_sequences_file=MagicMock(),
                mapping_file=MagicMock(),
            )

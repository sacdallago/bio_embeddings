from pathlib import Path
from typing import Type

import numpy
import pytest

import torch
import os


from bio_embeddings.embed import (
    SeqVecEmbedder,
    AlbertEmbedder,
    BertEmbedder,
    XLNetEmbedder,
    EmbedderInterface,
)

all_embedders = [SeqVecEmbedder, AlbertEmbedder, BertEmbedder, XLNetEmbedder]


def embedder_test_impl(embedder_class: Type[EmbedderInterface], use_cpu: bool):
    """ Compute embeddings and check them against a stored reference file """
    expected_file = Path("test-data/reference-embeddings").joinpath(
        embedder_class.name + ".npz"
    )

    expected = numpy.load(str(expected_file))
    # TODO: Change this once GH-20 is solved
    if os.environ.get("MODEL_DIRECTORY"):
        model_directory = Path(os.environ["MODEL_DIRECTORY"]).joinpath(
            embedder_class.name
        )
        embedder = embedder_class(model_directory=model_directory, use_cpu=use_cpu)
    else:
        embedder = embedder_class.with_download(use_cpu=use_cpu)
    [protein, seqwence] = embedder.embed_many(["PROTEIN", "SEQWENCE"])
    assert numpy.allclose(expected["test_case 1"], protein)
    assert numpy.allclose(expected["test_case 2"], seqwence)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Can't test the GPU if there isn't any"
)
@pytest.mark.parametrize("embedder_class", all_embedders)
def test_embedder_gpu(embedder_class: Type[EmbedderInterface]):
    embedder_test_impl(embedder_class, False)


@pytest.mark.parametrize("embedder_class", all_embedders)
def test_embedder_cpu(embedder_class: Type[EmbedderInterface]):
    embedder_test_impl(embedder_class, True)

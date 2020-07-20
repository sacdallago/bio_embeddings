from pathlib import Path
from typing import Type

import numpy
import pytest
import torch

from bio_embeddings.embed import (
    SeqVecEmbedder,
    AlbertEmbedder,
    ShortAlbertEmbedder,
    BertEmbedder,
    XLNetEmbedder,
    EmbedderInterface,
)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This is too slow on the CPU and for CI"
)
@pytest.mark.parametrize(
    "name,embedder_class",
    [
        ("seqvec", SeqVecEmbedder),
        ("short_albert", ShortAlbertEmbedder),
        ("albert", AlbertEmbedder),
        ("bert", BertEmbedder),
        ("xlnet", XLNetEmbedder),
    ],
)
def test_embedder(name: str, embedder_class: Type[EmbedderInterface]):
    expected_file = Path("test-data/reference-embeddings").joinpath(name + ".npz")
    if not expected_file.is_file():
        print("TODO: Get reference embeddings for all embedders")
        return

    expected = numpy.load(str(expected_file))
    embedder = embedder_class.with_download()
    [seqwence, protein] = embedder.embed_many(["SEQWENCE", "PROTEIN"])
    print(expected["SEQWENCE"].shape, seqwence.shape)
    print(expected["SEQWENCE"].mean(), seqwence.mean())
    assert numpy.allclose(expected["SEQWENCE"], seqwence)
    assert numpy.allclose(expected["PROTEIN"], protein)

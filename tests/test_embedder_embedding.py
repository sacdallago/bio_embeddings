"""
How to add a new embedder:

```
from bio_embeddings.embed import MyLanguageModel
import numpy
embedder = MyLanguageModel()
[protein, seqwence, padded] = embedder.embed_many(["PROTEIN", "SEQWENCE", "VLSXXXIEP"])
numpy.savez(f"test-data/reference-embeddings/{embedder.name}.npz", **{"test_case 1": protein, "test_case 2": seqwence})
```

Don't forget to add the weights in defaults.yml and to upload them
"""

import os
from pathlib import Path
from typing import Type, Optional, Any, List

import numpy
import pytest
import torch

from bio_embeddings.embed import (
    BeplerEmbedder,
    CPCProtEmbedder,
    ESM1bEmbedder,
    ESMEmbedder,
    EmbedderInterface,
    PLUSRNNEmbedder,
    ProtTransAlbertBFDEmbedder,
    ProtTransBertBFDEmbedder,
    ProtTransT5BFDEmbedder,
    ProtTransT5UniRef50Embedder,
    ProtTransT5XLU50Embedder,
    ProtTransXLNetUniRef100Embedder,
    SeqVecEmbedder,
    UniRepEmbedder,
    Word2VecEmbedder,
    FastTextEmbedder,
    GloveEmbedder,
    OneHotEncodingEmbedder,
    ESM1vEmbedder,
)
from bio_embeddings.project.pb_tucker import PBTucker
from bio_embeddings.utilities import get_model_file
from tests.shared import check_embedding

common_embedders: List[Any] = [
    BeplerEmbedder,
    CPCProtEmbedder,
    ESM1bEmbedder,
    ESM1vEmbedder,
    FastTextEmbedder,
    GloveEmbedder,
    OneHotEncodingEmbedder,
    PLUSRNNEmbedder,
    ProtTransBertBFDEmbedder,
    SeqVecEmbedder,
    Word2VecEmbedder,
]


# Those embedder aren't ran by default on CI
neglected_embedders: List[Any] = [
    ESMEmbedder,
    ProtTransAlbertBFDEmbedder,
    ProtTransT5BFDEmbedder,
    ProtTransT5UniRef50Embedder,
    ProtTransT5XLU50Embedder,
    ProtTransXLNetUniRef100Embedder,
]

all_embedders: List[Any] = common_embedders + [
    pytest.param(
        embedder_class,
        marks=pytest.mark.skipif(
            os.environ.get("SKIP_NEGLECTED_EMBEDDER_TESTS"), reason="Save CI resources"
        ),
    )
    for embedder_class in neglected_embedders
]


def embedder_test_impl(
    embedder_class: Type[EmbedderInterface], device: Optional[str] = None
):
    """Compute embeddings and check them against a stored reference file"""
    if embedder_class == SeqVecEmbedder:
        embedder = embedder_class(warmup_rounds=0, device=device)
    elif embedder_class == ESM1vEmbedder:
        embedder = embedder_class(ensemble_id=1, device=device)
    else:
        embedder = embedder_class(device=device)

    batch_size = 100

    # The XXX tests that the unknown padding works
    # https://github.com/sacdallago/bio_embeddings/issues/63
    padded_sequence = "VLSXXXIEP"
    [protein, seqwence, padded, _empty] = embedder.embed_many(
        ["PROTEIN", "SEQWENCE", padded_sequence, ""], batch_size
    )

    # Checks that the XXX has kept its length during embedding
    check_embedding(embedder, padded, padded_sequence)

    # Check with reference embeddings
    expected_file = Path("test-data/reference-embeddings").joinpath(
        embedder.name + ".npz"
    )
    expected = numpy.load(str(expected_file))
    assert numpy.allclose(expected["test_case 1"], protein, rtol=1.0e-3, atol=1.0e-5)
    assert numpy.allclose(expected["test_case 2"], seqwence, rtol=1.0e-3, atol=1.0e-5)


@pytest.mark.skipif(os.environ.get("SKIP_SLOW_TESTS"), reason="This test is very slow")
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Can't test the GPU if there isn't any"
)
@pytest.mark.parametrize("embedder_class", all_embedders)
def test_embedder_gpu(embedder_class: Type[EmbedderInterface]):
    embedder_test_impl(embedder_class, "cuda")


@pytest.mark.skipif(os.environ.get("SKIP_SLOW_TESTS"), reason="This test is very slow")
@pytest.mark.parametrize("embedder_class", all_embedders)
def test_embedder_cpu(embedder_class: Type[EmbedderInterface]):
    embedder_test_impl(embedder_class, "cpu")


@pytest.mark.skipif(os.environ.get("SKIP_SLOW_TESTS"), reason="This test is very slow")
@pytest.mark.skipif(os.environ.get("SKIP_AVX2_TESTS"), reason="This test needs AVX2")
@pytest.mark.parametrize("embedder_class", [UniRepEmbedder])
def test_embedder_other(embedder_class: Type[EmbedderInterface]):
    """UniRep does not allow configuring the device"""
    embedder_test_impl(embedder_class, None)


@pytest.mark.skipif(os.environ.get("SKIP_SLOW_TESTS"), reason="This test is very slow")
@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="Can't run CUDA tests without a GPU",
            ),
        ),
    ],
)
def test_tucker(pytestconfig, device):
    bert_embeddings_file = pytestconfig.rootpath.joinpath(
        "test-data/reference-embeddings"
    ).joinpath(ProtTransBertBFDEmbedder.name + ".npz")
    bert_embeddings = numpy.load(bert_embeddings_file)
    tucker_embeddings_file = pytestconfig.rootpath.joinpath(
        "test-data/reference-embeddings"
    ).joinpath(PBTucker.name + ".npz")
    tucker_embeddings = numpy.load(tucker_embeddings_file)

    pb_tucker = PBTucker(get_model_file("pb_tucker", "model_file"), device)

    for name, embedding in bert_embeddings.items():
        reduced_embedding = embedding.mean(axis=0)
        tucker_embedding = pb_tucker.project_reduced_embedding(reduced_embedding)
        assert numpy.allclose(
            tucker_embeddings[name], tucker_embedding, rtol=1.0e-3, atol=1.0e-5
        ), name

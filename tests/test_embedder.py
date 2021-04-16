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
import typing
from json import JSONDecodeError
from pathlib import Path
from typing import Optional
from typing import Type
from unittest import mock

import numpy
import pytest
import torch
from numpy import ndarray

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
    ProtTransT5XLU50,
    ProtTransXLNetUniRef100Embedder,
    SeqVecEmbedder,
    UniRepEmbedder,
)
from bio_embeddings.embed.pipeline import embed_and_write_batched
from bio_embeddings.embed.prottrans_t5_embedder import ProtTransT5Embedder
from bio_embeddings.project.pb_tucker import PBTucker
from bio_embeddings.utilities import read_fasta, FileSystemFileManager, get_model_file
from tests.shared import check_embedding

all_embedders = [
    BeplerEmbedder,
    CPCProtEmbedder,
    ESMEmbedder,
    ESM1bEmbedder,
    PLUSRNNEmbedder,
    ProtTransAlbertBFDEmbedder,
    ProtTransBertBFDEmbedder,
    pytest.param(
        ProtTransT5BFDEmbedder,
        marks=pytest.mark.skipif(
            not os.environ.get("TEST_OLD_T%"),
            reason="Those tests are slow and you should use prottrans_t5_xl_u50",
        ),
    ),
    pytest.param(
        ProtTransT5UniRef50Embedder,
        marks=pytest.mark.skipif(
            not os.environ.get("TEST_OLD_T%"),
            reason="Those tests are slow and you should use prottrans_t5_xl_u50",
        ),
    ),
    pytest.param(
        ProtTransT5XLU50,
        marks=pytest.mark.skipif(
            os.environ.get("SKIP_T5"), reason="T5 makes ci run out of disk"
        ),
    ),
    ProtTransXLNetUniRef100Embedder,
    SeqVecEmbedder,
]


def embedder_test_impl(
    embedder_class: Type[EmbedderInterface], device: Optional[str] = None
):
    """ Compute embeddings and check them against a stored reference file """
    if embedder_class == SeqVecEmbedder:
        embedder = embedder_class(warmup_rounds=0, device=device)
    else:
        embedder = embedder_class(device=device)

    if isinstance(embedder, ProtTransT5Embedder):
        batch_size = None
    else:
        batch_size = 100

    # The XXX tests that the unknown padding works
    # https://github.com/sacdallago/bio_embeddings/issues/63
    padded_sequence = "VLSXXXIEP"
    [protein, seqwence, padded] = embedder.embed_many(
        ["PROTEIN", "SEQWENCE", padded_sequence], batch_size
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
@pytest.mark.parametrize("embedder_class", [UniRepEmbedder])
def test_embedder_other(embedder_class: Type[EmbedderInterface]):
    """UniRep does not allow configuring the device"""
    embedder_test_impl(embedder_class, None)


@pytest.mark.parametrize(
    "embedder_class",
    [
        ProtTransAlbertBFDEmbedder,
        ProtTransBertBFDEmbedder,
        ProtTransXLNetUniRef100Embedder,
    ],
)
def test_model_download(embedder_class):
    """ We want to check that models are downloaded if the model_directory isn't given """
    module_name = embedder_class.__module__
    model_class = typing.get_type_hints(embedder_class)["_model"].__name__
    model_name = f"{module_name}.{model_class}"
    tokenizer_name = model_name.replace("Model", "Tokenizer")
    with mock.patch(
        "bio_embeddings.embed.embedder_interfaces.get_model_directories_from_zip",
        return_value="/dev/null",
    ) as get_model_mock, mock.patch(model_name, mock.MagicMock()), mock.patch(
        tokenizer_name, mock.MagicMock()
    ):
        embedder_class()
    get_model_mock.assert_called_once()


@pytest.mark.skipif(os.environ.get("SKIP_SLOW_TESTS"), reason="This test is very slow")
@pytest.mark.parametrize(
    "embedder_class",
    [
        ProtTransAlbertBFDEmbedder,
        ProtTransBertBFDEmbedder,
        ProtTransXLNetUniRef100Embedder,
    ],
)
def test_model_no_download(embedder_class):
    """ We want to check that models aren't downloaded if the model_directory is given """
    with mock.patch(
        "bio_embeddings.embed.embedder_interfaces.get_model_directories_from_zip",
        return_value="/dev/null",
    ) as get_model_mock:
        with pytest.raises(OSError):
            embedder_class(model_directory="/none/existent/path")
        get_model_mock.assert_not_called()


def test_model_parameters_seqvec(caplog):
    with mock.patch(
        "bio_embeddings.embed.embedder_interfaces.get_model_file",
        return_value="/dev/null",
    ) as get_model_mock:
        # Since we're not actually downloading, the json options file is empty
        with pytest.raises(JSONDecodeError):
            SeqVecEmbedder(weights_file="/none/existent/path")
    get_model_mock.assert_called_once()
    assert caplog.messages == [
        "You should pass either all necessary files or directories, or none, while "
        "you provide 1 of 2"
    ]

    with pytest.raises(FileNotFoundError):
        SeqVecEmbedder(model_directory="/none/existent/path")
    with pytest.raises(FileNotFoundError):
        SeqVecEmbedder(
            weights_file="/none/existent/path", options_file="/none/existent/path"
        )


@pytest.mark.skipif(os.environ.get("SKIP_T5"), reason="T5 makes ci run out of disk")
@pytest.mark.skipif(os.environ.get("SKIP_SLOW_TESTS"), reason="This test is very slow")
def test_batching_t5_blocked():
    """Once the T5 bug is fixed, this should become a regression test"""
    embedder = ProtTransT5BFDEmbedder()
    with pytest.raises(RuntimeError):
        embedder.embed_many([], batch_size=1000)


@pytest.mark.skipif(os.environ.get("SKIP_T5"), reason="T5 makes ci run out of disk")
@pytest.mark.skipif(os.environ.get("SKIP_SLOW_TESTS"), reason="This test is very slow")
def test_batching_t5(pytestconfig):
    """Check that T5 batching is still failing"""
    embedder = ProtTransT5BFDEmbedder()
    fasta_file = pytestconfig.rootpath.joinpath("examples/docker/fasta.fa")
    batch = [str(i.seq[:]) for i in read_fasta(str(fasta_file))]
    embeddings_single_sequence = list(
        super(ProtTransT5Embedder, embedder).embed_many(batch, batch_size=None)
    )
    embeddings_batched = list(
        super(ProtTransT5Embedder, embedder).embed_many(batch, batch_size=10000)
    )
    for a, b in zip(embeddings_single_sequence, embeddings_batched):
        assert not numpy.allclose(a, b) and numpy.allclose(
            a, b, rtol=1.0e-4, atol=1.0e-5
        )


def test_half_precision_embedder(pytestconfig, caplog, tmp_path: Path):
    """Currently a dummy test"""

    class Float16Embedder(EmbedderInterface):
        name = "float16embedder"
        embedding_dimension = 1024
        number_of_layers = 1

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            assert kwargs.get("half_precision_model"), kwargs

        def embed(self, sequence: str) -> ndarray:
            return numpy.random.random((len(sequence), 1024)).astype(numpy.float16)

        @staticmethod
        def reduce_per_protein(embedding: ndarray) -> ndarray:
            return embedding.sum(axis=0)

    result_kwargs = {
        "prefix": str(tmp_path),
        "remapped_sequences_file": str(
            pytestconfig.rootpath.joinpath("test-data/remapped_sequences_file.fasta")
        ),
        "mapping_file": str(
            pytestconfig.rootpath.joinpath("test-data/mapping_file.csv")
        ),
        "half_precision_model": True,
    }
    embed_and_write_batched(
        Float16Embedder(**result_kwargs),
        FileSystemFileManager(),
        result_kwargs=result_kwargs,
    )

    assert caplog.messages == []


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

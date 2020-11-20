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

from bio_embeddings.embed import (
    BeplerEmbedder,
    CPCProtEmbedder,
    ESMEmbedder,
    EmbedderInterface,
    PLUSRNNEmbedder,
    ProtTransAlbertBFDEmbedder,
    ProtTransBertBFDEmbedder,
    ProtTransXLNetUniRef100Embedder,
    SeqVecEmbedder,
    UniRepEmbedder,
)

all_embedders = [
    BeplerEmbedder,
    CPCProtEmbedder,
    ESMEmbedder,
    PLUSRNNEmbedder,
    ProtTransAlbertBFDEmbedder,
    ProtTransBertBFDEmbedder,
    ProtTransXLNetUniRef100Embedder,
    SeqVecEmbedder,
]


def embedder_test_impl(
    embedder_class: Type[EmbedderInterface], device: Optional[str] = None
):
    """ Compute embeddings and check them against a stored reference file """
    expected_file = Path("test-data/reference-embeddings").joinpath(
        embedder_class.name + ".npz"
    )

    if embedder_class == SeqVecEmbedder:
        embedder = embedder_class(warmup_rounds=0, device=device)
    else:
        embedder = embedder_class(device=device)
    # The XXX tests that the unknown padding works
    # https://github.com/sacdallago/bio_embeddings/issues/63
    padded_sequence = "VLSXXXIEP"
    [protein, seqwence, padded] = embedder.embed_many(
        ["PROTEIN", "SEQWENCE", padded_sequence], 100
    )

    # Check that the XXX has kept its length during embedding
    if embedder_class == SeqVecEmbedder:
        assert padded.shape[1] == len(padded_sequence)
    elif embedder_class == UniRepEmbedder:
        # Not sure why this is one longer, but the jax-unirep tests check
        # `len(sequence) + 1`, so it seems to be intended
        assert padded.shape[0] == len(padded_sequence) + 1
    elif embedder_class == CPCProtEmbedder:
        # There is only a per-protein embedding for CPCProt
        assert padded.shape == (512,)
    else:
        assert padded.shape[0] == len(padded_sequence)

    # Check reduce_per_protein
    # https://github.com/sacdallago/bio_embeddings/issues/85
    assert embedder.reduce_per_protein(protein).shape == (embedder.embedding_dimension,)

    # Check with reference embeddings
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

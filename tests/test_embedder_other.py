import os
import typing
from json import JSONDecodeError
from pathlib import Path
from unittest import mock

import numpy
import pytest
from numpy import ndarray

from bio_embeddings.embed import (
    ESM1bEmbedder,
    EmbedderInterface,
    ProtTransAlbertBFDEmbedder,
    ProtTransBertBFDEmbedder,
    ProtTransXLNetUniRef100Embedder,
    SeqVecEmbedder,
    name_to_embedder,
)
from bio_embeddings.embed.pipeline import embed_and_write_batched
from bio_embeddings.utilities import FileSystemFileManager
from bio_embeddings.utilities.config import read_config_file


@pytest.mark.parametrize(
    "embedder_class",
    [
        ProtTransAlbertBFDEmbedder,
        ProtTransBertBFDEmbedder,
        ProtTransXLNetUniRef100Embedder,
    ],
)
def test_model_download(embedder_class):
    """We want to check that models are downloaded if the model_directory isn't given"""
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
    """We want to check that models aren't downloaded if the model_directory is given"""
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
def test_esm_max_len():
    embedder = ESM1bEmbedder(device="cpu")
    message = "esm1b only allows sequences up to 1022 residues, but your longest sequence is 2021 residues long"
    with pytest.raises(ValueError, match=message):
        embedder.embed("M" * 2021)
    with pytest.raises(ValueError, match=message):
        list(embedder.embed_batch(["SEQWENCE", "M" * 2021, "PROTEIN"]))
    with pytest.raises(ValueError, match=message):
        list(embedder.embed_many(["SEQWENCE", "M" * 2021, "PROTEIN"], batch_size=5000))


def test_model_sizes_for_all_embedder(pytestconfig):
    """Make sure we have the model sizes documented for each model

    If this test is failing, run the following and enter the results in bio_embeddings/embed/__init__.py:

    ```
    python -m bio_embeddings.utilities.model_size_main cpu
    python -m bio_embeddings.utilities.model_size_main gpu
    ```
    """
    models = read_config_file(
        pytestconfig.rootpath.joinpath("bio_embeddings/utilities/defaults.yml")
    )
    set(models.keys())
    doc_text: str = pytestconfig.rootpath.joinpath(
        "bio_embeddings/embed/__init__.py"
    ).read_text()
    # Quick and stupid rst parsing
    documented_embedder = set()
    for line in doc_text.split("=" * 46)[2].splitlines()[1:]:
        documented_embedder.add(line.split(" ")[0])
    assert name_to_embedder.keys() - set(documented_embedder) == set()
    # Handle the non-embedder models
    assert set(documented_embedder) - name_to_embedder.keys() == {
        "bert_from_publication",
        "deepblast",
        "pb_tucker",
        "seqvec_from_publication",
    }

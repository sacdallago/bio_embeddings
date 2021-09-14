import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Any, Dict
from unittest import mock
from unittest.mock import MagicMock

import numpy
import pytest
from numpy import ndarray

from bio_embeddings.embed import name_to_embedder, ESM1bEmbedder
from bio_embeddings.embed.pipeline import run, DEFAULT_MAX_AMINO_ACIDS
from bio_embeddings.utilities import InvalidParameterError
from bio_embeddings.utilities.config import read_config_file
from bio_embeddings.utilities.pipeline import execute_pipeline_from_config


class MockElmoEmbedder:
    # noinspection PyUnusedLocal
    def __init__(self, options_file: str, weight_file: str, cuda_device: int = -1):
        self.embeddedings = numpy.load("test-data/embeddings.npz")

    def embed_batch(self, many: List[str]) -> List[ndarray]:
        return [self.embed_sentence(i) for i in many]

    def embed_sentence(self, _sentence: str) -> ndarray:
        return list(self.embeddedings.values())[0]  # TODO


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


class MockESM1bEmbedder(ESM1bEmbedder):
    # noinspection PyMissingConstructor
    def __init__(self, **kwargs):
        pass

    def embed_batch(self, many: List[str]) -> List[ndarray]:
        return [self.embed_sentence(i) for i in many]

    def embed_sentence(self, _sentence: str) -> ndarray:
        return numpy.random.random((self.number_of_layers, self.embedding_dimension))


def read_and_patch_config(
    pytestconfig, tmp_path: Path, yml_file: str
) -> Dict[str, Any]:
    pipeline_config = read_config_file(str(pytestconfig.rootpath.joinpath(yml_file)))
    pipeline_config["global"]["sequences_file"] = str(
        pytestconfig.rootpath.joinpath("test-data").joinpath(
            pipeline_config["global"]["sequences_file"]
        )
    )
    pipeline_config["global"]["prefix"] = str(
        tmp_path.joinpath(pipeline_config["global"]["prefix"])
    )
    return pipeline_config


def test_wrong_model_param(pytestconfig, tmp_path: Path, caplog):
    """In this config, the protocol esm1b is chosen, but instead of a model_file a model_directory for T5 is given"""
    pipeline_config = read_and_patch_config(
        pytestconfig, tmp_path, "test-data/embed_config_mixup.yml"
    )

    with mock.patch(
        "bio_embeddings.embed.pipeline.name_to_embedder", {"esm1b": MockESM1bEmbedder}
    ), mock.patch(
        "bio_embeddings.embed.embedder_interfaces.get_model_file",
        return_value="/dev/null",
    ):
        execute_pipeline_from_config(pipeline_config)

    assert caplog.messages == [
        "You set an unknown option for esm1b: model_directory (value: /mnt/project/bio_embeddings/models/lms/t5)"
    ]


def test_esm1v_missing_ensemble_id(pytestconfig, tmp_path: Path):
    pipeline_config = read_and_patch_config(
        pytestconfig, tmp_path, "test-data/esm1v_missing_ensemble_id.yml"
    )
    with pytest.raises(
        InvalidParameterError,
        match=re.escape(
            "You must set `ensemble_id` to select which of the five models you want to use [1-5]"
        ),
    ):
        execute_pipeline_from_config(pipeline_config)

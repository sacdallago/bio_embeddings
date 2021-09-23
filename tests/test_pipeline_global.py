import re
from pathlib import Path

import importlib_metadata
import pytest
import toml
from packaging import version

from bio_embeddings.utilities import InvalidParameterError
from bio_embeddings.utilities.pipeline import (
    execute_pipeline_from_config,
    parse_config_file_and_execute_run,
)


def test_pipeline_global(tmp_path):
    prefix = Path(tmp_path).joinpath("prefix")
    out_config = execute_pipeline_from_config(
        {
            "global": {
                "prefix": str(prefix),
                "sequences_file": "test-data/seqwence-protein.fasta",
            }
        }
    )

    try:
        installed_version = version.parse(importlib_metadata.version("bio_embeddings"))
        expected = version.parse(
            toml.loads(Path("pyproject.toml").read_text())["tool"]["poetry"]["version"]
        )
        # That can actually happen
        assert expected == installed_version, "Please run `poetry install`"
        print(out_config["global"])
        assert version.parse(out_config["global"]["version"]) == expected
    except importlib_metadata.PackageNotFoundError:
        pass  # No dev install

    expected_files = [
        "input_parameters_file.yml",
        "mapping_file.csv",
        "output_parameters_file.yml",
        "remapped_sequences_file.fasta",
        "sequences_file.fasta",
    ]

    assert sorted(expected_files) == sorted(path.name for path in prefix.iterdir())


def test_no_config_file():
    with pytest.raises(
        InvalidParameterError,
        match="The configuration file at '/no/such/path' does not exist",
    ):
        parse_config_file_and_execute_run("/no/such/path")


def test_invalid_config_file():
    with pytest.raises(
        InvalidParameterError,
        match=re.escape(
            f"Could not parse configuration file at '{__file__}' as yaml. "
            f"Formatting mistake in config file? See Error above for details."
        ),
    ):
        parse_config_file_and_execute_run(__file__)

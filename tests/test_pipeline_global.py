from pathlib import Path

import importlib_metadata
import toml

from bio_embeddings.utilities.pipeline import execute_pipeline_from_config


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
        installed_version = importlib_metadata.version("bio_embeddings")
        expected = toml.loads(Path("pyproject.toml").read_text())["tool"]["poetry"][
            "version"
        ]
        # That can actually happen
        assert expected == installed_version, "Please run `poetry install`"
        print(out_config["global"])
        assert out_config["global"]["version"] == expected
    except importlib_metadata.PackageNotFoundError:
        pass  # No dev install

    expected_files = [
        "input_parameters_file.yml",
        "mapping_file.csv",
        "ouput_parameters_file.yml",
        "remapped_sequences_file.fasta",
        "sequences_file.fasta",
    ]

    assert sorted(expected_files) == sorted(path.name for path in prefix.iterdir())

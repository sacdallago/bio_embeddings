from pathlib import Path

import importlib_metadata
import toml

from bio_embeddings.utilities.pipeline import execute_pipeline_from_config


def test_pipeline_global(tmp_path):
    prefix = Path(tmp_path).joinpath("prefix")
    execute_pipeline_from_config(
        {
            "global": {
                "prefix": str(prefix),
                "sequences_file": "test-data/seqwence-protein.fasta",
            }
        }
    )

    expected_files = [
        "input_parameters_file.yml",
        "mapping_file.csv",
        "ouput_parameters_file.yml",
        "remapped_sequences_file.fasta",
        "sequences_file.fasta",
    ]

    try:
        importlib_metadata.version("bio_embeddings")
        actual = prefix.joinpath("bio_embeddings_version.txt").read_text()
        expected = toml.loads(Path("pyproject.toml").read_text())["tool"]["poetry"][
            "version"
        ]
        assert actual == expected
        expected_files.append("bio_embeddings_version.txt")
    except importlib_metadata.PackageNotFoundError:
        pass  # No dev install

    assert sorted(expected_files) == sorted(path.name for path in prefix.iterdir())

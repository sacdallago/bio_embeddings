from pathlib import Path

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

    actual = prefix.joinpath("bio_embeddings_version.txt").read_text()
    expected = toml.loads(Path("pyproject.toml").read_text())["tool"]["poetry"][
        "version"
    ]
    assert actual == expected

    expected_files = [
        "bio_embeddings_version.txt",
        "input_parameters_file.yml",
        "mapping_file.csv",
        "ouput_parameters_file.yml",
        "remapped_sequences_file.fasta",
        "sequences_file.fasta",
    ]
    assert expected_files == sorted(path.name for path in prefix.iterdir())

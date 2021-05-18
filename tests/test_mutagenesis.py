import os
from pathlib import Path

import pytest

from bio_embeddings.mutagenesis.pipeline import run


@pytest.mark.skipif(
    not os.environ.get("RUN_VERY_SLOW_TESTS"),
    reason="This ",
)
def test_protbert_bfd_mutagenesis(pytestconfig, tmp_path: Path):
    run(
        protocol="protbert_bfd_mutagenesis",
        prefix=tmp_path,
        stage_name="protbert_bfd_mutagenesis_test",
        remapped_sequences_file=str(
            pytestconfig.rootpath.joinpath(
                "test-data/remapped_sequences_file.fasta"
            )
        ),
        mapping_file=str(
            pytestconfig.rootpath.joinpath("test-data/mapping_file.csv")
        ),
    )

    assert tmp_path.joinpath(
        "protbert_bfd_mutagenesis_test/78c685273a9456e98046482c09b31473.html"
    ).is_file()
    assert tmp_path.joinpath(
        "protbert_bfd_mutagenesis_test/a75fa3e22c2a164e8d5632867b4e2dd8.html"
    ).is_file()

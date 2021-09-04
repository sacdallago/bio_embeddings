import os
from pathlib import Path

import pytest

from bio_embeddings.mutagenesis.pipeline import run as run_mutagenesis
from bio_embeddings.visualize.pipeline import run as run_visualize


@pytest.mark.skipif(
    not os.environ.get("RUN_VERY_SLOW_TESTS"), reason="This is experimental"
)
def test_protbert_bfd_mutagenesis(pytestconfig, tmp_path: Path):
    result_kwargs = run_mutagenesis(
        protocol="protbert_bfd_mutagenesis",
        prefix=tmp_path,
        stage_name="protbert_bfd_mutagenesis_test",
        remapped_sequences_file=str(
            pytestconfig.rootpath.joinpath("test-data/remapped_sequences_file.fasta")
        ),
        mapping_file=str(pytestconfig.rootpath.joinpath("test-data/mapping_file.csv")),
    )

    stage_parameters = dict(
        stage_name="plot_mutagenesis_test",
        type="visualize",
        protocol="plot_mutagenesis",
        depends_on="mutagenesis",
    )

    run_visualize(**{**result_kwargs, **stage_parameters})

    assert tmp_path.joinpath(
        "plot_mutagenesis_test/78c685273a9456e98046482c09b31473.html"
    ).is_file()
    assert tmp_path.joinpath(
        "plot_mutagenesis_test/a75fa3e22c2a164e8d5632867b4e2dd8.html"
    ).is_file()

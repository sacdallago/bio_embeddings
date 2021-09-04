import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from bio_embeddings.align import (
    check_mmseqs, convert_mmseqs_result_to_profile,
    create_mmseqs_database, mmseqs_search, MMseqsSearchOptions,
    MMseqsSearchOptionsEnum
)


@pytest.mark.skipif(os.environ.get("SKIP_MMSEQS_TESTS"), reason="Skip MMseqs2 tests.")
def test_mmseqs_installed():
    assert check_mmseqs(), True


@pytest.mark.skipif(not check_mmseqs(), reason="mmseqs2 binary not found")
def test_basic_mmseqs2():
    sequence_search_options = MMseqsSearchOptions()
    sequence_search_options.add_option(MMseqsSearchOptionsEnum.alignment_output, True)
    sequence_search_options.add_option(MMseqsSearchOptionsEnum.num_iterations, 3)

    profile_search_options = MMseqsSearchOptions()
    profile_search_options.add_option(MMseqsSearchOptionsEnum.minimum_sequence_identity, .2)
    profile_search_options.add_option(MMseqsSearchOptionsEnum.sensitivity, 7.5)
    profile_search_options.add_option(MMseqsSearchOptionsEnum.maximum_number_of_return_sequences, 1000)

    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        create_mmseqs_database(Path("test-data/remapped_sequences_file.fasta"), temp_dir_path/"query")
        create_mmseqs_database(Path("test-data/subcellular_location_new_hard_set.fasta"), temp_dir_path/"search")

        # Sequence to sequence search
        mmseqs_search(temp_dir_path/"query", temp_dir_path/"search", temp_dir_path/"result", sequence_search_options)

        # Convert sequence-to-sequence results to profile
        convert_mmseqs_result_to_profile(
            temp_dir_path/"query", temp_dir_path/"search", temp_dir_path/"result", temp_dir_path/"profile"
        )

        # Sequence to profile search
        mmseqs_search(
            temp_dir_path / "query", temp_dir_path / "profile", temp_dir_path / "profile_result",
            profile_search_options
        )


# TODO: Do some kind of check of output integrity.


@pytest.mark.skipif(not check_mmseqs(), reason="mmseqs2 binary not found")
def test_mmseqs_outputs():
    pass

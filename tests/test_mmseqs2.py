import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from bio_embeddings.align import (
    check_mmseqs, convert_mmseqs_result_to_profile,
    create_mmseqs_database, mmseqs_search, MMseqsSearchOptions,
    MMseqsSearchOptionsEnum, convert_result_to_alignment_file
)


@pytest.mark.skipif(not check_mmseqs(), reason="mmseqs2 binary not found")
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
    profile_search_options.add_option(MMseqsSearchOptionsEnum.maximum_number_of_prefilter_sequences, 1000)
    profile_search_options.add_option(MMseqsSearchOptionsEnum.alignment_output, True)

    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        query_dir = temp_dir_path/"query"
        query_dir.mkdir()
        serch_dir = temp_dir_path/"search"
        serch_dir.mkdir()
        result_dir = temp_dir_path/"result"
        result_dir.mkdir()

        create_mmseqs_database(Path("test-data/remapped_sequences_file.fasta"), query_dir)
        create_mmseqs_database(Path("test-data/subcellular_location_new_hard_set.fasta"), serch_dir)

        # Sequence to sequence search
        mmseqs_search(query_dir, serch_dir, result_dir, sequence_search_options)

        profile_dir = temp_dir_path/"profile"
        profile_dir.mkdir()
        profile_result_dir = temp_dir_path/"result_profile"
        profile_result_dir.mkdir()

        # Convert sequence-to-sequence results to profile
        convert_mmseqs_result_to_profile(
            query_dir, serch_dir, result_dir, profile_dir
        )

        # Sequence to profile search
        mmseqs_search(
            query_dir, profile_dir, profile_result_dir,
            profile_search_options
        )

        convert_result_to_alignment_file(
            query_dir,
            profile_dir,
            profile_result_dir,
            temp_dir_path / "alignment.tsv"
        )


# TODO: Do some kind of check of output integrity.


@pytest.mark.skipif(not check_mmseqs(), reason="mmseqs2 binary not found")
def test_mmseqs_outputs():
    pass

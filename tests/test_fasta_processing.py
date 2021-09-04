from pathlib import Path

import numpy
import pytest
from numpy import ndarray

from bio_embeddings.embed import EmbedderInterface
from bio_embeddings.embed.pipeline import embed_and_write_batched
from bio_embeddings.utilities.filemanagers import FileSystemFileManager

# noinspection PyProtectedMember
from bio_embeddings.utilities.pipeline import _process_fasta_file


class FakeEmbedder(EmbedderInterface):
    embedding_dimension = 1024
    number_of_layers = 1

    def embed(self, sequence: str) -> ndarray:
        return numpy.asarray([])

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        return embedding


def test_simple_remapping(pytestconfig, tmp_path: Path):
    """https://github.com/sacdallago/bio_embeddings/issues/50"""
    global_parameters = {
        "sequences_file": str(
            pytestconfig.rootpath.joinpath("test-data/seqwence-protein.fasta")
        ),
        "prefix": str(tmp_path),
        "simple_remapping": True,
    }
    global_parameters = _process_fasta_file(**global_parameters)
    embed_and_write_batched(FakeEmbedder(), FileSystemFileManager(), global_parameters)


def test_illegal_amino_acids(caplog, pytestconfig, tmp_path: Path):
    """https://github.com/sacdallago/bio_embeddings/issues/54"""
    input_file = pytestconfig.rootpath.joinpath("test-data/illegal_amino_acids.fasta")
    _process_fasta_file(sequences_file=str(input_file), prefix=str(tmp_path))
    assert caplog.messages == [
        f"The entry 'lowercase' in {input_file} contains lower "
        "case amino acids. Lower case letters are uninterpretable by most language "
        "models, and their embedding will be nonesensical. Protein LMs available "
        "through bio_embeddings have been trained on upper case, single letter code "
        "sequence representations only "
        "(https://en.wikipedia.org/wiki/Amino_acid#Table_of_standard_amino_acid_abbreviations_and_properties)."
    ]


def test_unparsable_fasta(caplog, pytestconfig, tmp_path: Path):
    input_file = pytestconfig.rootpath.joinpath("test-data/unparsable.fasta")

    with pytest.raises(
        ValueError,
        match=f"Could not parse '{input_file}'. Are you sure this is a valid fasta file?",
    ):
        _process_fasta_file(sequences_file=str(input_file), prefix=str(tmp_path))
    assert caplog.messages == []


def test_broken_fasta(pytestconfig, tmp_path: Path):
    """Ensure that we print a reasonable message when the user feeds in a broken fasta file.

    Unfortunately, we can't detect if the user fed in a markdown file
    instead of a fasta, because we could parse that markdown file
    as fasta:

    > Following the initial line (used for a unique description of the
    > sequence) was the actual sequence itself in standard one-letter
    > character string. Anything other than a valid character would
    > be ignored (including spaces, tabulators, asterisks, etc...).
    > It was also common to end the sequence with an "*" (asterisk)
    > character (in analogy with use in PIR formatted sequences) and,
    > for the same reason, to leave a blank line between the description
    > and the sequence.

    From https://www.wikiwand.com/en/FASTA_format

    NCBI is a bit stricter with their definition by barring blank line,
    but is otherwise still extremely lenient.
    (https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=BlastHelp)
    """
    input_file = pytestconfig.rootpath.joinpath("test-data/embeddings.npz")
    with pytest.raises(ValueError, match="Are you sure this is a valid fasta file?"):
        _process_fasta_file(sequences_file=input_file, prefix=str(tmp_path))


def test_missing_fasta(tmp_path: Path):
    input_file = tmp_path.joinpath("non_existant.fasta")
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        _process_fasta_file(sequences_file=input_file, prefix=str(tmp_path))

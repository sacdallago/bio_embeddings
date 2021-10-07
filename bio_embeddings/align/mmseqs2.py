import logging
import time

from typing import List, Any, Dict
from tempfile import TemporaryDirectory
from subprocess import check_call
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class MMseqsSearchOptionsEnum(Enum):
    sensitivity = "-s"
    num_iterations = "--num-iterations"
    e_value_cutoff = "-e"
    alignment_output = "-a"
    minimum_sequence_identity = "--min-seq-id"
    maximum_number_of_prefilter_sequences = "--max-seqs"

    @staticmethod
    def from_str(option_name: str) -> Enum:
        option = {
            "sensitivity": MMseqsSearchOptionsEnum.sensitivity,
            "num_iterations": MMseqsSearchOptionsEnum.num_iterations,
            "e_value_cutoff": MMseqsSearchOptionsEnum.e_value_cutoff,
            "alignment_output": MMseqsSearchOptionsEnum.alignment_output,
            "minimum_sequence_identity": MMseqsSearchOptionsEnum.minimum_sequence_identity,
            "maximum_number_of_prefilter_sequences": MMseqsSearchOptionsEnum.maximum_number_of_prefilter_sequences,
        }.get(option_name, None)

        if not option:
            raise Exception(f"Invalid option {option_name}.")

        return option


class MMseqsSearchOptions:
    _option_types = {
        MMseqsSearchOptionsEnum.sensitivity: float,
        MMseqsSearchOptionsEnum.num_iterations: int,
        MMseqsSearchOptionsEnum.e_value_cutoff: float,
        MMseqsSearchOptionsEnum.alignment_output: bool,
        MMseqsSearchOptionsEnum.minimum_sequence_identity: float,
        MMseqsSearchOptionsEnum.maximum_number_of_prefilter_sequences: int
    }

    _options: Dict[MMseqsSearchOptionsEnum, Any]

    def __init__(self):
        self._options = dict()

    def add_option(self, option: MMseqsSearchOptionsEnum, value: Any):
        # Assert value type is what it needs to be
        # Pycharm complains here but it's ok the way it is
        if not isinstance(value, self._option_types[option]):
            raise TypeError(f"Option {option.name} is of type {self._option_types[option]}, but you passed {value}.")

        self._options[option] = value

    def has_option(self, option: MMseqsSearchOptionsEnum) -> bool:
        return option in self._options.keys()

    def get_options(self) -> List[str]:
        result = []

        for option in self._options.keys():
            if option == MMseqsSearchOptionsEnum.alignment_output:
                continue
            else:
                value = self._options[option]
                result.append(option.value)
                result.append(str(value))

        produce_alignment = self._options.get(MMseqsSearchOptionsEnum.alignment_output, False)

        if produce_alignment:
            result.append(MMseqsSearchOptionsEnum.alignment_output.value)

        return result


def check_mmseqs() -> bool:
    try:
        mmseqs_path = check_call(["which", "mmseqs"])
        logger.info(f"mmseqs binary identified at: {mmseqs_path}")
        return True
    except OSError:
        return False


def create_mmseqs_database(fasta_file: Path, database_name: Path):
    """
    Will raise:
     - CalledProcessError if non-0 exit
     - OSError if executable is not found
    """
    database_name.parent.mkdir(exist_ok=True, parents=True)
    check_call(["mmseqs", "createdb", str(fasta_file), str(database_name / "sequence_database")])


_DEFAULT_MMSEQS_OPTIONS = MMseqsSearchOptions()


def mmseqs_search(
        query_database: Path,
        search_database: Path,
        search_result_directory: Path,
        search_options: MMseqsSearchOptions = _DEFAULT_MMSEQS_OPTIONS
) -> None:
    """Calls `mmseqs search`"""

    logger.info("Searching with MMseqs2")
    start = time.time()
    # Otherwise MMseqs2 will complain that "result_mmseqs2.dbtype exists already"
    for old_result_file in search_result_directory.glob("*"):
        old_result_file.unlink()
    # usage: mmseqs search <i:queryDB> <i:targetDB> <o:alignmentDB> <tmpDir> [options]
    with TemporaryDirectory() as temp_dir:
        check_call(
            [
                "mmseqs",
                "search",
                str(query_database / "sequence_database"),
                str(search_database / "sequence_database"),
                str(search_result_directory / "search_results"),
                temp_dir,
                *search_options.get_options()
            ]
        )
    total = time.time() - start
    logger.info(f"`mmseqs search` took {total :f}s")


def convert_mmseqs_result_to_profile(
        query_database: Path,
        search_database: Path,
        search_result_directory: Path,
        profile_directory: Path
) -> None:
    check_call(
        [
            "mmseqs",
            "result2profile",
            str(query_database / "sequence_database"),
            str(search_database / "sequence_database"),
            str(search_result_directory / "search_results"),
            str(profile_directory / "sequence_database")
        ]
    )


def convert_result_to_alignment_file(
        query_database: Path,
        search_database: Path,
        search_result_directory: Path,
        search_result_file: Path
) -> None:
    """
    Output format: TSV
    Columns: query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,pident,nident,qlen,
             tlen,qcov,tcov,qaln,taln
    """
    check_call(
        [
            "mmseqs",
            "convertalis",
            str(query_database / "sequence_database"),
            str(search_database / "sequence_database"),
            str(search_result_directory / "search_results"),
            str(search_result_file),
            "--format-output",
            "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,pident,nident,qlen,"
            "tlen,qcov,tcov,qaln,taln"
        ]
    )

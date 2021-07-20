import logging
import os
import string
import sys
import traceback
import urllib.parse
from copy import deepcopy
from datetime import datetime
from typing import Dict, Callable, Optional, Any
from urllib import request

import importlib_metadata
import torch
from atomicwrites import atomic_write
from importlib_metadata import PackageNotFoundError

from bio_embeddings.embed.pipeline import run as run_embed
from bio_embeddings.extract.pipeline import run as run_extract
from bio_embeddings.mutagenesis.pipeline import run as run_mutagenesis
from bio_embeddings.project.pipeline import run as run_project
from bio_embeddings.utilities import get_file_manager, read_fasta, reindex_sequences, write_fasta_file, \
    check_required, MD5ClashException, InvalidParameterError
from bio_embeddings.utilities.config import read_config_file, write_config_file
from bio_embeddings.utilities.filemanagers import FileManagerInterface
from bio_embeddings.utilities.remote_file_retriever import TqdmUpTo
from bio_embeddings.visualize.pipeline import run as run_visualize

logger = logging.getLogger(__name__)

_STAGES = {
    "embed": run_embed,
    "project": run_project,
    "visualize": run_visualize,
    "extract": run_extract,
    "mutagenesis": run_mutagenesis,
}

try:
    # noinspection PyUnresolvedReferences
    from bio_embeddings.align.pipeline import run as run_align

    _STAGES["align"] = run_align
except ImportError as e:
    if not str(e).startswith("No module named 'deepblast"):
        raise
    else:
        def error(**kwargs):
            raise RuntimeError(
                "The extra for the deepblast protocol is missing. "
                "See https://docs.bioembeddings.com/#installation on how to install all extras"
            )


        _STAGES["align"] = error

_IN_CONFIG_NAME = "input_parameters_file"
_OUT_CONFIG_NAME = "ouput_parameters_file"

_ISSUE_URL = "https://github.com/sacdallago/bio_embeddings/issues/new"
_ERROR_REPORTING_TEMPLATE = """## Metadata
|key|value|
|--|--|
|**version**|{}|
|**cuda**|{}|

## Parameter
|key|value|
|--|--|
{}

## Traceback
```
{}```

## More info
"""


def _validate_file(file_path: str):
    """
    Verify if a file exists and is not empty.
    Parameters
    ----------
    file_path : str
        Path to file to check
    Returns
    -------
    bool
        True if file exists and is non-zero size,
        False otherwise.
    """
    try:
        if os.stat(file_path).st_size == 0:
            raise InvalidParameterError(f"The file at '{file_path}' is empty")
    except (OSError, TypeError) as e:
        raise InvalidParameterError(f"The configuration file at '{file_path}' does not exist") from e


def _process_fasta_file(**kwargs):
    """
    Will assign MD5 hash as ID if no if provided for a sequence.
    """
    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager(**kwargs)

    sequences = read_fasta(kwargs['sequences_file'])

    # Sanity check the fasta file to avoid nonsense and/or crashes by the embedders
    letters = set(string.ascii_letters)
    for entry in sequences:
        illegal = sorted(set(entry.seq) - letters)
        if illegal:
            formatted = "'" + "', '".join(illegal) + "'"
            raise ValueError(
                f"The entry '{entry.name}' in {kwargs['sequences_file']} contains the characters {formatted}, "
                f"while only single letter code is allowed "
                f"(https://en.wikipedia.org/wiki/Amino_acid#Table_of_standard_amino_acid_abbreviations_and_properties)."
            )
        # This is a warning due to the inconsistent handling between different embedders
        if not str(entry.seq).isupper():
            logger.warning(
                f"The entry '{entry.name}' in {kwargs['sequences_file']} contains lower case amino acids. "
                f"Lower case letters are uninterpretable by most language models, "
                f"and their embedding will be nonesensical. "
                f"Protein LMs available through bio_embeddings have been trained on upper case, "
                f"single letter code sequence representations only "
                f"(https://en.wikipedia.org/wiki/Amino_acid#Table_of_standard_amino_acid_abbreviations_and_properties)."
            )

    sequences_file_path = file_manager.create_file(kwargs.get('prefix'), None, 'sequences_file',
                                                   extension='.fasta')
    write_fasta_file(sequences, sequences_file_path)

    result_kwargs['sequences_file'] = sequences_file_path

    # Remap using sequence position rather than md5 hash -- not encouraged!
    result_kwargs['simple_remapping'] = result_kwargs.get('simple_remapping', False)

    mapping = reindex_sequences(sequences, simple=result_kwargs['simple_remapping'])

    # Check if there's the same MD5 index twice. This most likely indicates 100% sequence identity.
    # Throw an error for MD5 hash clashes!
    if mapping.index.has_duplicates:
        raise MD5ClashException("There is at least one MD5 hash clash.\n"
                                "This most likely indicates there are multiple identical sequences in your FASTA file.\n"
                                "MD5 hashes are used to remap sequence identifiers from the input FASTA.\n"
                                "This error exists to prevent wasting resources (computing the same embedding twice).\n"
                                "There's a (very) low probability of this indicating a real MD5 clash.\n\n"
                                "If you are sure there are no identical sequences in your set, please open an issue at "
                                "https://github.com/sacdallago/bio_embeddings/issues . "
                                "Otherwise, use cd-hit to reduce your input FASTA to exclude identical sequences!")

    mapping_file_path = file_manager.create_file(kwargs.get('prefix'), None, 'mapping_file', extension='.csv')
    remapped_sequence_file_path = file_manager.create_file(kwargs.get('prefix'), None, 'remapped_sequences_file',
                                                           extension='.fasta')

    write_fasta_file(sequences, remapped_sequence_file_path)
    mapping.to_csv(mapping_file_path)

    result_kwargs['mapping_file'] = mapping_file_path
    result_kwargs['remapped_sequences_file'] = remapped_sequence_file_path

    return result_kwargs


def _null_function(config: Dict) -> None:
    pass


def download_files_for_stage(
    stage_parameters: Dict[str, Any],
    file_manager: FileManagerInterface,
    prefix: str,
    stage_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Download files given as url

    We don't actually check whether a given option actually takes a path, e.g.
    one could specify `simple_remapping: https://google.com` and we'd download
    google.com. It will however still be evident for the user that the error
    lies in `simple_remapping` even though we replaced the value
    """
    for key in stage_parameters:
        if isinstance(stage_parameters[key], str) and (
            stage_parameters[key].startswith("http://")
            or stage_parameters[key].startswith("https://")
            or stage_parameters[key].startswith("ftp://")
        ):
            filename = file_manager.create_file(prefix, stage_name, key)
            logger.info(f"Downloading {stage_parameters[key]} to {filename}")
            desc = stage_parameters[key].split("/")[-1]
            with atomic_write(filename, overwrite=True) as temp_file:
                with TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=desc) as tqdm:
                    request.urlretrieve(
                        stage_parameters[key], filename=temp_file.name, reporthook=tqdm.update_to
                    )
            stage_parameters[key] = filename
    return stage_parameters


def execute_pipeline_from_config(config: Dict,
                                 post_stage: Callable[[Dict], None] = _null_function,
                                 **kwargs) -> Dict:
    original_config = deepcopy(config)

    check_required(
        config,
        ["global"]
    )

    # !! pop = remove from config!
    global_parameters = config.pop('global')

    check_required(
        global_parameters,
        ["prefix", "sequences_file"]
    )

    file_manager = get_file_manager(**global_parameters)

    # Make sure prefix exists
    prefix = global_parameters['prefix']

    # If prefix already exists
    if file_manager.exists(prefix):
        if not kwargs.get('overwrite'):
            raise FileExistsError("The prefix already exists & no overwrite option has been set.\n"
                                  "Either set --overwrite, or move data from the prefix.\n"
                                  "Prefix: {}".format(prefix))
    else:
        # create the prefix
        file_manager.create_prefix(prefix)

    # Copy original config to prefix
    global_in = file_manager.create_file(prefix, None, _IN_CONFIG_NAME, extension='.yml')
    write_config_file(global_in, original_config)

    # This downloads sequences_file if required
    download_files_for_stage(global_parameters, file_manager, prefix)

    global_parameters = _process_fasta_file(**global_parameters)

    for stage_name in config:
        stage_parameters = config[stage_name]
        original_stage_parameters = dict(**stage_parameters)

        check_required(
            stage_parameters,
            ["protocol", "type"]
        )

        stage_type = stage_parameters['type']
        stage_runnable = _STAGES.get(stage_type)

        if not stage_runnable:
            raise Exception("No type defined, or invalid stage type defined: {}".format(stage_type))

        # Prepare to run stage
        stage_parameters['stage_name'] = stage_name
        file_manager.create_stage(prefix, stage_name)

        stage_parameters = download_files_for_stage(stage_parameters, file_manager, prefix, stage_name)

        stage_dependency = stage_parameters.get('depends_on')

        if stage_dependency:
            if stage_dependency not in config:
                raise Exception("Stage {} depends on {}, but dependency not found in config.".format(stage_name,
                                                                                                     stage_dependency))

            stage_dependency_parameters = config.get(stage_dependency)
            stage_parameters = {**global_parameters, **stage_dependency_parameters, **stage_parameters}
        else:
            stage_parameters = {**global_parameters, **stage_parameters}

        # Register start time
        start_time = datetime.now().astimezone()
        stage_parameters['start_time'] = str(start_time)

        stage_in = file_manager.create_file(prefix, stage_name, _IN_CONFIG_NAME, extension='.yml')
        write_config_file(stage_in, stage_parameters)

        try:
            stage_output_parameters = stage_runnable(**stage_parameters)
        except Exception as e:
            # We are pretty sure that it's wrong configuration and not a bug
            if isinstance(e, InvalidParameterError):
                raise

            # Tell the user which stage failed and show an url to report an error on github
            try:
                version = importlib_metadata.version("bio_embeddings")
            except PackageNotFoundError:
                version = "unknown"

            # Make a github flavored markdown table; the header is in the template
            parameter_table = "\n".join(
                f"{key}|{value}" for key, value in original_stage_parameters.items()
            )
            params = {
                # https://stackoverflow.com/a/35498685/3549270
                "title": f"Protocol {original_stage_parameters['protocol']}: {type(e).__name__}: {e}",
                "body": _ERROR_REPORTING_TEMPLATE.format(
                    version,
                    torch.cuda.is_available(),
                    parameter_table,
                    traceback.format_exc(10),
                ),
            }
            print(traceback.format_exc(), file=sys.stderr)
            print(
                f"Consider reporting this error at this url: {_ISSUE_URL}?{urllib.parse.urlencode(params)}\n\n"
                f"Stage {stage_name} failed.",
                file=sys.stderr,
            )

            sys.exit(1)

        # Register end time
        end_time = datetime.now().astimezone()
        stage_output_parameters['end_time'] = str(end_time)

        # Register elapsed time
        stage_output_parameters['elapsed_time'] = str(end_time - start_time)

        stage_out = file_manager.create_file(prefix, stage_name, _OUT_CONFIG_NAME, extension='.yml')
        write_config_file(stage_out, stage_output_parameters)

        # Store in global_out config for later retrieval (e.g. depends_on)
        config[stage_name] = stage_output_parameters

        # Execute post-stage function, if provided
        post_stage(stage_output_parameters)

    config['global'] = global_parameters

    try:
        config['global']['version'] = importlib_metadata.version("bio_embeddings")
    except PackageNotFoundError:
        pass  # :(

    global_out = file_manager.create_file(prefix, None, _OUT_CONFIG_NAME, extension='.yml')
    write_config_file(global_out, config)

    return config


def parse_config_file_and_execute_run(config_file_path: str, **kwargs):
    _validate_file(config_file_path)

    # read configuration and execute
    config = read_config_file(config_file_path)

    execute_pipeline_from_config(config, **kwargs)

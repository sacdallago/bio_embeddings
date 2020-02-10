import os
from copy import deepcopy

from bio_embeddings.embed.pipeline import run as run_embed
# from bio_embeddings.extract_features.pipeline import run as run_extract_features
from bio_embeddings.utilities import get_file_manager, read_fasta_file, reindex_sequences, write_fasta_file, \
    check_required
from bio_embeddings.utilities.config import read_config_file

_STAGES = {
    "embed": run_embed,
    # "extract_features": run_extract_features
}


def _valid_file(file_path):
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
        return os.stat(file_path).st_size > 0
    except (OSError, TypeError):
        # catch TypeError for nonsense paths, e.g. None
        return False


def _process_fasta_file(**kwargs):
    result_kwargs = deepcopy(kwargs)
    _FILE_MANAGER = get_file_manager(**kwargs)

    sequences = read_fasta_file(kwargs['sequences_file'])
    mapping = reindex_sequences(sequences)

    # TODO: _STAGE_NAME none for global?
    mapping_file_path = _FILE_MANAGER.create_file(kwargs.get('prefix'), _STAGE_NAME, 'mapping_file', extension='.csv')
    remapped_sequence_file_path = _FILE_MANAGER.create_file(kwargs.get('prefix'), _STAGE_NAME, 'remapped_sequences_file'
                                                            , extension='.fasta')

    write_fasta_file(sequences, remapped_sequence_file_path)
    mapping.to_csv(mapping_file_path)

    result_kwargs['mapping_file'] = mapping_file_path
    result_kwargs['remapped_sequences_file'] = remapped_sequence_file_path

    return result_kwargs


def run(config_file_path):

    if not _valid_file(config_file_path):
        raise Exception("No config or invalid config was passed.")

    # read configuration and execute
    config = read_config_file(config_file_path)

    check_required(
        config,
        ["global"]
    )

    global_parameters = config.pop('global')

    check_required(
        global_parameters,
        ["prefix", "sequences_file"]
    )

    file_manager = get_file_manager(**global_parameters)

    # Make sure prefix exists
    prefix = global_parameters['prefix']
    file_manager.create_prefix(prefix)

    global_parameters *= _process_fasta_file(**global_parameters)

    for stage_name in config:
        stage_parameters = config[stage_name]

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

        stage_dependency = stage_parameters.get('depends_on')

        if stage_dependency and stage_dependency not in config:
            raise Exception("Stage {} depends on {}, but dependency not found in config.".format(stage_name,
                                                                                                 stage_dependency))

        stage_dependency_parameters = config.get(stage_dependency)

        stage_parameters = {**global_parameters, **stage_dependency_parameters, **stage_parameters}

        stage_output_parameters = stage_runnable(stage_parameters)
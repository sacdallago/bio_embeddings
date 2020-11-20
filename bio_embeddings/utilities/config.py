"""
Configuration handling

Authors:
  Thomas A. Hopf
"""
from pathlib import Path
from typing import Union

import ruamel.yaml as yaml
from ruamel.yaml.parser import ParserError
from ruamel.yaml.comments import CommentedBase

from bio_embeddings.utilities.exceptions import InvalidParameterError


def parse_config(config_str: str, preserve_order: bool = True) -> dict:
    """
    Parse a configuration string

    Parameters
    ----------
    config_str : str
        Configuration to be parsed
    preserve_order : bool, optional (default: False)
        Preserve formatting of input configuration
        string

    Returns
    -------
    dict
        Configuration dictionary
    """
    try:
        if preserve_order:
            return yaml.load(config_str, Loader=yaml.RoundTripLoader)
        else:
            return yaml.safe_load(config_str)
    except ParserError as e:
        raise InvalidParameterError(
            "Could not parse input configuration. "
            "Formatting mistake in config file? "
            "See ParserError above for details."
        ) from e


def read_config_file(config_path: Union[str, Path], preserve_order: bool = True) -> dict:
    """
    Read config from path to file.

    :param config_path: path to .yml config file
    :param preserve_order:
    :return:
    """
    with open(config_path, "r") as f:
        return parse_config(f.read(), preserve_order)


def write_config_file(out_filename: str, config: dict) -> None:
    """
    Save configuration data structure in YAML file.

    Parameters
    ----------
    out_filename : str
        Filename of output file
    config : dict
        Config data that will be written to file
    """
    if isinstance(config, CommentedBase):
        dumper = yaml.RoundTripDumper
    else:
        dumper = yaml.Dumper

    with open(out_filename, "w") as f:
        f.write(
            yaml.dump(config, Dumper=dumper, default_flow_style=False)
        )

"""
Configuration handling

Authors:
  Thomas A. Hopf
"""
from pathlib import Path
from typing import Union

from ruamel import yaml
from ruamel.yaml import YAMLError
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
    except YAMLError as e:
        raise InvalidParameterError(
            f"Could not parse configuration file at {config_str} as yaml. "
            "Formatting mistake in config file? "
            "See Error above for details."
        ) from e


def read_config_file(config_path: Union[str, Path], preserve_order: bool = True) -> dict:
    """
    Read config from path to file.

    :param config_path: path to .yml config file
    :param preserve_order:
    :return:
    """
    with open(config_path, "r") as fp:
        try:
            if preserve_order:
                return yaml.load(fp, Loader=yaml.RoundTripLoader)
            else:
                return yaml.safe_load(fp)
        except YAMLError as e:
            raise InvalidParameterError(
                f"Could not parse configuration file at '{config_path}' as yaml. "
                "Formatting mistake in config file? "
                "See Error above for details."
            ) from e


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

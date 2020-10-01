#!/usr/bin/env python3

import argparse
import logging

from bio_embeddings.utilities.pipeline import parse_config_file_and_execute_run


def main():
    """
    Pipeline commandline entry point
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Jax likes to print warnings
    logging.captureWarnings(True)

    parser = argparse.ArgumentParser(description='Embeds protein sequences.')
    parser.add_argument('-o', '--overwrite', dest='overwrite', required=False, action='store_true',
                        help='Will force overwrite of previously existing results.')
    parser.add_argument('config_path', metavar='/path/to/pipeline_definition.yml', type=str, nargs=1,
                        help='The path to the config. For examples, see folder "parameter examples".')
    arguments = parser.parse_args()

    parse_config_file_and_execute_run(arguments.config_path[0], overwrite=arguments.overwrite)


if __name__ == '__main__':
    main()

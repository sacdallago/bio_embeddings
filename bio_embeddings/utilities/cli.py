#!/bin/python

import argparse
import logging

from bio_embeddings.utilities.pipeline import run


def main():
    """
    Pipeline commandline entry point
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description='Embeds protein sequences.')
    parser.add_argument('-o', '--overwrite', dest='overwrite', required=False, action='store_true',
                        help='Will force overwrite of previously existing results.')
    parser.add_argument('config_path', metavar='/path/to/pipeline_definition.yml', type=str, nargs=1,
                        help='The path to the config. For examples, see folder "parameter examples".')
    arguments = parser.parse_args()

    run(arguments.config_path[0], overwrite=arguments.overwrite)


if __name__ == '__main__':
    main()

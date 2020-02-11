#!/bin/python

import argparse
from bio_embeddings.utilities.pipeline import run

parser = argparse.ArgumentParser(description='Embeds protein sequences.')

parser.add_argument('--overwrite', action='store', dest='overwrite', required=False, default=False, const=True,
                    help='Will force overwrite of previously existing results.')
parser.add_argument('config_path', metavar='/path/to/config.yml', type=str, nargs=1, required=True,
                    help='The path to the config. For examples, see folder "parameter examples".')

arguments = parser.parse_args()

if __name__ == '__main__':
    run(arguments.config_path, overwrite=arguments.overwrite)

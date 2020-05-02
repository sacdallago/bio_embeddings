#!/bin/python

import argparse
from bio_embeddings.utilities.pipeline import run

parser = argparse.ArgumentParser(description='Embeds protein sequences.')

parser.add_argument('-o', '--overwrite', dest='overwrite', required=False, action='store_true',
                    help='Will force overwrite of previously existing results.')
parser.add_argument('config_path', metavar='/path/to/simple.yml', type=str, nargs=1,
                    help='The path to the config. For examples, see folder "parameter examples".')

arguments = parser.parse_args()


if __name__ == '__main__':
    run(arguments.config_path[0], overwrite=arguments.overwrite)

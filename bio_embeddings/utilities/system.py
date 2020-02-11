#!/bin/python

import argparse
from bio_embeddings.utilities.pipeline import run

parser = argparse.ArgumentParser(
    description='Embeds protein sequences and stores their embeddings as np array.')

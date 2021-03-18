#!/bin/bash
# Unlike what you'll want to do in production, we want to check the h5 into the repository
# and want to run the two parts separately
bio_embeddings knn_reference_embed.yml -o
python make_reference.py
bio_embeddings gopredsim.yml -o

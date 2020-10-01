#!/bin/bash

# Make the source available for the webserver by including the root of the repo
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH="{$DIR}/..":$PYTHONPATH

# For more information about environment variables to set, check out the webserver.configuration.configuration file

# These are needed by both celery workers and backend
export CELERY_BROKER_URL="amqp://localhost"
export MONGO_URL="mongodb://localhost:27017"

# The following are not needed for the backend to function! -- only for the celery workers!
export BERT_MODEL_DIRECTORY=""
export BERT_SECONDARY_STRUCTURE_CHECKPOINT_FILE=""
export BERT_SUBCELLULAR_LOCATION_CHECKPOINT_FILE=""
export SEQVEC_WEIGHTS_FILE=""
export SEQVEC_OPTIONS_FILE=""
export SEQVEC_SECONDARY_STRUCTURE_CHECKPOINT_FILE=""
export SEQVEC_SUBCELLULAR_LOCATION_CHECKPOINT_FILE=""

python webserver/backend.py

exit 0
#!/bin/bash

# Make the source available for the webserver by including the root of the repo
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH="{$DIR}/..":$PYTHONPATH

# For more information about environment variables to set, check out the webserver.configuration.configuration file

# These are needed by both celery workers and backend
export CELERY_BROKER_URL="amqp://localhost"
export MONGO_URL="mongodb://localhost:27017"

# The following needed for the celery workers!
export BERT_MODEL_DIRECTORY="/mnt/nfs/models/bert"
export BERT_SECONDARY_STRUCTURE_CHECKPOINT_FILE="/mnt/nfs/models/bert/secondary_structure_checkpoint_file"
export BERT_SUBCELLULAR_LOCATION_CHECKPOINT_FILE="/mnt/nfs/models/bert/subcellular_location_checkpoint_file"
export SEQVEC_WEIGHTS_FILE="/mnt/nfs/models/seqvec/weights_file"
export SEQVEC_OPTIONS_FILE="/mnt/nfs/models/seqvec/options_file"
export SEQVEC_SECONDARY_STRUCTURE_CHECKPOINT_FILE="/mnt/nfs/models/seqvec/secondary_structure_checkpoint_file"
export SEQVEC_SUBCELLULAR_LOCATION_CHECKPOINT_FILE="/mnt/nfs/models/seqvec/subcellular_location_checkpoint_file"

celery worker -A celery_worker.task_keeper --loglevel=info --pool=solo

exit 0
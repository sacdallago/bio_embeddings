#!/bin/bash

# Make the source available for the webserver by including the root of the repo
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH="{$DIR}/..":$PYTHONPATH

# For more information about environment variables to set, check out the webserver.configuration.configuration file

# These are needed by both celery workers and backend
export CELERY_BROKER_URL="amqp://localhost"
export MONGO_URL="mongodb://root:example@localhost:27017"

# The following needed for the celery workers!
export MODEL_DIRECTORY="/mnt/nfs/models"
# Optional, if you want to create a "sync" worker
# Values can be: seqvec, seqvec_annotations, protbert, protbert_annotations or (default) pipeline
# export CELERY_WORKER_TYPE="seqvec"

# Important parameters are
#  - Q: which queues to run
#  - n: name of the host (if running on the same machine, should be different!)

celery worker -A celery_worker.task_keeper --loglevel=info --pool=solo -n pipeline_worker -Q pipeline

exit 0

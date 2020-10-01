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

celery worker -A celery_worker.task_keeper --loglevel=info --pool=solo

exit 0

#!/bin/bash

export CELERY_BROKER_URL="amqp://localhost"
export PYTHONPATH=.:$PYTHONPATH
celery worker -A webserver.celery_worker.task_keeper --loglevel=info --pool=solo

exit 0
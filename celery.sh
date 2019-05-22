#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH
celery worker -A webserver.celery_worker.task_keeper --loglevel=info --pool=solo

exit 0
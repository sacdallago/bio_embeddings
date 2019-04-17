#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH
celery worker -A app.celery_worker.task_keeper --loglevel=info --pool=solo

exit 0
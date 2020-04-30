#!/bin/bash

export CELERY_BROKER_URL="amqp://localhost"
export PYTHONPATH=.:$PYTHONPATH
python webserver/backend.py

exit 0
from os import environ
from celery import Celery as _celery

_broker = environ['CELERY_BROKER_URL']

task_keeper = _celery(broker=_broker, backend=_broker)

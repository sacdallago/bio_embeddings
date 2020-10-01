from webserver.utilities.configuration import configuration
from celery import Celery as _celery

task_keeper = _celery(broker=configuration['web']['celery_broker_url'],
                      backend=configuration['web']['celery_broker_url'])
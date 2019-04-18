from os import environ
from celery import Celery as _celery
import sys

_broker = environ['CELERY_BROKER_URL']

task_keeper = _celery(broker=_broker, backend=_broker)

# Default task expiry time.
# Set to 5 minutes (60s * 5)
task_keeper.expires = 60*5

IN_CELERY_WOKER_PROCESS = False

if len(sys.argv) > 0 and sys.argv[0].endswith('celery')\
        and 'worker' in sys.argv:
    IN_CELERY_WOKER_PROCESS=True
    print ('This is a Celery worker')

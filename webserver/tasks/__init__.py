from webserver.utilities.configuration import configuration
from celery import Celery as _celery

task_keeper = _celery('tasks',
    broker=configuration["web"]["celery_broker_url"], backend="rpc://"
)

task_keeper.conf.task_routes = {
    'webserver.tasks.prott5_embeddings.get_prott5_embeddings_sync': {'queue': 'prott5'},
    'webserver.tasks.prott5_annotations.get_prott5_annotations_sync': {'queue': 'prott5_annotations'},
    'webserver.tasks.pipeline.run_pipeline': {'queue': 'pipeline'},
    'webserver.tasks.vespa_pred.get_vespa_output_sync':{'queue':'vespa'}
}

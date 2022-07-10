from webserver.utilities.configuration import configuration
from celery import Celery as _celery

task_keeper = _celery('tasks',
    broker=configuration["web"]["celery_broker_url"], backend="rpc://"
)

task_keeper.conf.task_routes = {
    'webserver.tasks.prott5_embeddings.get_prott5_embeddings_sync': {'queue': 'prott5'},
    'webserver.tasks.prott5_annotations.get_prott5_annotations_sync': {'queue': 'prott5_annotations'},
    'webserver.tasks.colabfold.get_structure_colabfold': {'queue': 'colabfold'},
    'webserver.tasks.pipeline.run_pipeline': {'queue': 'pipeline'},
    'webserver.tasks.prott5_residue_landscape_annotations.get_residue_landscape_output_sync':{'queue':'prott5_residue_landscape_annotations'}
}

from webserver.tasks import task_keeper
from webserver.utilities.configuration import configuration

# Make celery find the task dependent on the worker type
if "prott5" in configuration['celery']['celery_worker_type']:
    # noinspection PyUnresolvedReferences
    from webserver.tasks.prott5_embeddings import get_prott5_embeddings_sync
if "prott5_annotations" in configuration['celery']['celery_worker_type']:
    # noinspection PyUnresolvedReferences
    from webserver.tasks.prott5_annotations import get_prott5_annotations_sync
if "prott5_residue_landscape_annotations" in configuration['celery']['celery_worker_type']:
    # add ann extra worker to compute the residue landscape
    from webserver.tasks.prott5_residue_landscape_annotations import get_residue_landscape_output_sync
if "colabfold" in configuration['celery']['celery_worker_type']:
    from webserver.tasks.colabfold import get_structure_colabfold
if "pipeline" in configuration['celery']['celery_worker_type']:
    # noinspection PyUnresolvedReferences
    from webserver.tasks.pipeline import run_pipeline


if __name__ == "__main__":
    task_keeper.start()

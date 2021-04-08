from webserver.tasks import task_keeper
from webserver.utilities.configuration import configuration

# Make celery find the task dependent on the worker type
if "seqvec" in configuration['celery']['celery_worker_type']:
    # noinspection PyUnresolvedReferences
    from webserver.tasks.seqvec_embeddings import get_seqvec_embeddings_sync
if "seqvec_annotations" in configuration['celery']['celery_worker_type']:
    # noinspection PyUnresolvedReferences
    from webserver.tasks.seqvec_annotations import get_seqvec_annotations_sync
if "protbert" in configuration['celery']['celery_worker_type']:
    # noinspection PyUnresolvedReferences
    from webserver.tasks.protbert_embeddings import get_protbert_embeddings_sync
if "protbert_annotations" in configuration['celery']['celery_worker_type']:
    # noinspection PyUnresolvedReferences
    from webserver.tasks.protbert_annotations import get_protbert_annotations_sync
if "prott5" in configuration['celery']['celery_worker_type']:
    # noinspection PyUnresolvedReferences
    from webserver.tasks.prott5_embeddings import get_prott5_embeddings_sync
if "prott5_annotations" in configuration['celery']['celery_worker_type']:
    # noinspection PyUnresolvedReferences
    from webserver.tasks.prott5_annotations import get_prott5_annotations_sync
if "pipeline" in configuration['celery']['celery_worker_type']:
    # noinspection PyUnresolvedReferences
    from webserver.tasks.pipeline import run_pipeline

if __name__ == "__main__":
    task_keeper.start()


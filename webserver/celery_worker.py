from webserver.utilities.configuration import configuration
from webserver.tasks import task_keeper

# Make celery find the task dependent on the worker type
if configuration['celery']['celery_worker_type'] == "seqvec":
    # noinspection PyUnresolvedReferences
    from webserver.tasks.seqvec_embeddings import get_seqvec_embeddings_sync
elif configuration['celery']['celery_worker_type'] == "seqvec_annotations":
    # noinspection PyUnresolvedReferences
    from webserver.tasks.seqvec_annotations import get_seqvec_annotations_sync
elif configuration['celery']['celery_worker_type'] == "protbert":
    # noinspection PyUnresolvedReferences
    from webserver.tasks.protbert_embeddings import get_protbert_embeddings_sync
elif configuration['celery']['celery_worker_type'] == "protbert_annotations":
    # noinspection PyUnresolvedReferences
    from webserver.tasks.protbert_annotations import get_protbert_annotations_sync
else:
    # noinspection PyUnresolvedReferences
    from webserver.tasks.pipeline import run_pipeline

if __name__ == "__main__":
    task_keeper.start()

from webserver.utilities.configuration import configuration
from webserver.tasks import task_keeper

# Make celery find the task dependent on the worker type
if configuration['celery']['celery_worker_type'] == "seqvec" or configuration['celery']['celery_worker_type'] == "protbert":
    # noinspection PyUnresolvedReferences
    from webserver.tasks.embeddings import get_annotations_sync, get_embedding_sync
else:
    # noinspection PyUnresolvedReferences
    from webserver.tasks.embeddings import run_pipeline

if __name__ == "__main__":
    task_keeper.start()

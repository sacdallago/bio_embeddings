from webserver.tasks import task_keeper

# Make celery find the task
# noinspection PyUnresolvedReferences
from webserver.tasks.embeddings import get_embeddings

if __name__ == "__main__":
    task_keeper.start()

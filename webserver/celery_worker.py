from webserver.tasks import task_keeper
from webserver.tasks.embeddings import get_embeddings

if __name__ == '__main__':
    task_keeper.start()

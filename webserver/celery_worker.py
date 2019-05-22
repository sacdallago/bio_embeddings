from webserver.tasks import task_keeper
from webserver.machine_learning import get_seqvec

if __name__ == '__main__':
    task_keeper.start()

import logging
from typing import Dict

from webserver.tasks import task_keeper

logger = logging.getLogger()


@task_keeper.task()
def get_structure_colabfold(sequence: str) -> Dict[str, str]:
    logger.info('Calling get_structure_colabfold')
    return {
        'sequence': sequence,
        'structure': 'TODO'
    }

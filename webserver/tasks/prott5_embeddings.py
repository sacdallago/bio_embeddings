import logging

from typing import List

from webserver.tasks import task_keeper
from webserver.utilities.configuration import configuration

logger = logging.getLogger(__name__)

model = None

if "prott5" in configuration['celery']['celery_worker_type']:
    from bio_embeddings.embed import ProtTransT5BFDEmbedder

    logger.info("Loading the language model...")

    model = ProtTransT5BFDEmbedder(
        model_directory=configuration['prottrans_t5_bfd']['model_directory'],
        decoder=False,
        half_precision_model=False
    )

    logger.info("Finished initializing.")


@task_keeper.task()
def get_prott5_embeddings_sync(sequence: str) -> List:
    return model.embed(sequence).tolist()

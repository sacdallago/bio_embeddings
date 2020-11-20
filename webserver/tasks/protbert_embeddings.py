import logging

from typing import List

from webserver.tasks import task_keeper
from webserver.utilities.configuration import configuration

logger = logging.getLogger(__name__)

model = None

if "protbert" in configuration['celery']['celery_worker_type']:
    from bio_embeddings.embed import ProtTransBertBFDEmbedder

    logger.info("Loading the language model...")

    model = ProtTransBertBFDEmbedder(
        model_directory=configuration['prottrans_bert_bfd']['model_directory']
    )

    logger.info("Finished initializing.")


@task_keeper.task()
def get_protbert_embeddings_sync(sequence: str) -> List:
    return model.embed(sequence).tolist()

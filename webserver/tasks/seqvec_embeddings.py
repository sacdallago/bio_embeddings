import logging

from typing import List

from webserver.tasks import task_keeper
from webserver.utilities.configuration import configuration

logger = logging.getLogger(__name__)

model = None

if "seqvec" in configuration['celery']['celery_worker_type']:
    from bio_embeddings.embed import SeqVecEmbedder

    logger.info("Loading the language model...")

    model = SeqVecEmbedder(
        options_file=configuration['seqvec']['options_file'],
        weights_file=configuration['seqvec']['weights_file']
    )

    logger.info("Finished initializing.")


@task_keeper.task()
def get_seqvec_embeddings_sync(sequence: str) -> List:
    return model.embed(sequence).tolist()

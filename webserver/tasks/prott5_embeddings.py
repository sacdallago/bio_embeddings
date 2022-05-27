import logging

from typing import List

from webserver.tasks import task_keeper
from webserver.utilities.configuration import configuration

logger = logging.getLogger(__name__)

model = None

if "prott5" in configuration['celery']['celery_worker_type']:
    import torch
    from bio_embeddings.embed import ProtTransT5XLU50Embedder

    logger.info("Loading the language model...")

    if torch.cuda.is_available():
        model = ProtTransT5XLU50Embedder(
            half_precision_model_directory=configuration['prottrans_t5_xl_u50']['half_model_directory'],
            half_precision_model=True,
            decoder=False
        )
    else:
        model = ProtTransT5XLU50Embedder(
            model_directory=configuration['prottrans_t5_xl_u50']['half_model_directory'],
            decoder=False,
        )

    logger.info("Finished initializing.")


@task_keeper.task()
def get_prott5_embeddings_sync(sequence: str) -> List:
    return model.embed(sequence).tolist()

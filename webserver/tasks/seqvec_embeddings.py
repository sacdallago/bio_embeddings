import logging

from webserver.tasks import task_keeper
from webserver.utilities.configuration import configuration

logger = logging.getLogger(__name__)

model = None
featureExtractor = None

if configuration['celery']['celery_worker_type'] == "seqvec":
    from bio_embeddings.embed import SeqVecEmbedder
    from bio_embeddings.extract.basic.BasicAnnotationExtractor import BasicAnnotationExtractor

    logger.info("Loading the language model...")

    model = SeqVecEmbedder(
        options_file=configuration['seqvec']['options_file'],
        weights_file=configuration['seqvec']['weights_file']
    )

    logger.info("Loading the feature extraction models...")

    featureExtractor = BasicAnnotationExtractor(
        "seqvec_from_publication",
        secondary_structure_checkpoint_file=configuration['seqvec']['secondary_structure_checkpoint_file'],
        subcellular_location_checkpoint_file=configuration['seqvec']['subcellular_location_checkpoint_file']
    )

    logger.info("Finished initializing.")


@task_keeper.task()
def get_seqvec_embeddings_sync(sequence: str):
    return model.embed(sequence)


@task_keeper.task()
def get_seqvec_annotations_sync(sequence: str):
    embedding = model.embed(sequence)
    annotations = featureExtractor.get_annotations(embedding)

    return annotations

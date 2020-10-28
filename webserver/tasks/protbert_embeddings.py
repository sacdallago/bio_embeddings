import logging

from webserver.tasks import task_keeper
from webserver.utilities.configuration import configuration

logger = logging.getLogger(__name__)

model = None
featureExtractor = None

if configuration['celery']['celery_worker_type'] == "protbert":
    from bio_embeddings.embed import ProtTransBertBFDEmbedder
    from bio_embeddings.extract.basic.BasicAnnotationExtractor import BasicAnnotationExtractor
    from bio_embeddings.utilities import convert_list_of_enum_to_string

    logger.info("Loading the language model...")

    model = ProtTransBertBFDEmbedder(
        model_directory=configuration['prottrans_bert_bfd']['model_directory']
    )

    logger.info("Loading the feature extraction models...")

    featureExtractor = BasicAnnotationExtractor(
        "bert_from_publication",
        secondary_structure_checkpoint_file=configuration['prottrans_bert_bfd']['secondary_structure_checkpoint_file'],
        subcellular_location_checkpoint_file=configuration['prottrans_bert_bfd']['subcellular_location_checkpoint_file']
    )

    logger.info("Finished initializing.")


@task_keeper.task()
def get_protbert_embeddings_sync(sequence: str):
    return model.embed(sequence).tolist()


@task_keeper.task()
def get_protbert_annotations_sync(sequence: str):
    embedding = model.embed(sequence)
    annotations = featureExtractor.get_annotations(embedding)

    return {
        "predictedDSSP3": convert_list_of_enum_to_string(annotations.DSSP3),
        "predictedDSSP8": convert_list_of_enum_to_string(annotations.DSSP8),
        "predictedDisorder": convert_list_of_enum_to_string(annotations.disorder),
        "predictedMembrane": annotations.membrane.value,
        "predictedSubcellularLocalizations": annotations.localization.value,
    }

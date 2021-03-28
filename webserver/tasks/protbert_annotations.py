import logging
import numpy as np

from typing import Dict

from webserver.tasks import task_keeper
from webserver.utilities.configuration import configuration

logger = logging.getLogger(__name__)

featureExtractor = None
la = None

if "protbert_annotations" in configuration['celery']['celery_worker_type']:
    from bio_embeddings.extract.basic.BasicAnnotationExtractor import BasicAnnotationExtractor
    from bio_embeddings.extract.light_attention import LightAttentionAnnotationExtractor

    from bio_embeddings.utilities import convert_list_of_enum_to_string

    logger.info("Loading the feature extraction models...")

    featureExtractor = BasicAnnotationExtractor(
        "bert_from_publication",
        secondary_structure_checkpoint_file=configuration['prottrans_bert_bfd']['secondary_structure_checkpoint_file'],
        subcellular_location_checkpoint_file=configuration['prottrans_bert_bfd']['subcellular_location_checkpoint_file']
    )

    print(configuration['prottrans_bert_bfd']['la_solubility_checkpoint_file'],
          configuration['prottrans_bert_bfd']['la_subcellular_location_checkpoint_file'])

    la = LightAttentionAnnotationExtractor(
        membrane_checkpoint_file=configuration['prottrans_bert_bfd']['la_solubility_checkpoint_file'],
        subcellular_location_checkpoint_file=configuration['prottrans_bert_bfd']['la_subcellular_location_checkpoint_file']
    )

    logger.info("Finished initializing.")


@task_keeper.task()
def get_protbert_annotations_sync(embedding: np.array) -> Dict[str, str]:
    annotations = featureExtractor.get_annotations(embedding)
    la_annotations = la.get_subcellular_location(embedding)

    return {
        "predictedDSSP3": convert_list_of_enum_to_string(annotations.DSSP3),
        "predictedDSSP8": convert_list_of_enum_to_string(annotations.DSSP8),
        "predictedDisorder": convert_list_of_enum_to_string(annotations.disorder),
        "predictedMembrane": la_annotations.membrane.value,
        "predictedSubcellularLocalizations": la_annotations.localization.value,
        "predictedCCO": [],
        "predictedBPO": [],
        "predictedMFO": []
    }

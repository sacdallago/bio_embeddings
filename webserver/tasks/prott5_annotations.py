import logging
import numpy as np

from typing import Dict

from webserver.tasks import task_keeper
from webserver.utilities.configuration import configuration

logger = logging.getLogger(__name__)

la = None

if "prott5_annotations" in configuration['celery']['celery_worker_type']:
    from bio_embeddings.extract.light_attention import LightAttentionAnnotationExtractor

    logger.info("Loading the feature extraction models...")

    la = LightAttentionAnnotationExtractor(
        membrane_checkpoint_file=configuration['prottrans_t5_bfd']['la_solubility_checkpoint_file'],
        subcellular_location_checkpoint_file=configuration['prottrans_t5_bfd']['la_subcellular_location_checkpoint_file']
    )

    logger.info("Finished initializing.")


@task_keeper.task()
def get_prott5_annotations_sync(embedding: np.array) -> Dict[str, str]:
    la_annotations = la.get_subcellular_location(embedding)

    return {
        "predictedMembrane": la_annotations.membrane.value,
        "predictedSubcellularLocalizations": la_annotations.localization.value,
        "predictedDSSP3": "",
        "predictedDSSP8": "",
        "predictedDisorder": "",
        "predictedCCO": [],
        "predictedBPO": [],
        "predictedMFO": [],
        "meta": {
            "predictedDSSP3": "unavailable",
            "predictedDSSP8": "unavailable",
            "predictedDisorder": "unavailable",
            "predictedCCO": "unavailable",
            "predictedBPO": "unavailable",
            "predictedMFO": "unavailable",
            "predictedMembrane": "LA_ProtT5",
            "predictedSubcellularLocalizations": "LA_ProtT5",
        }
    }

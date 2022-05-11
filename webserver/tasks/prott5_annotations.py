import logging
import numpy as np

from typing import Dict, List

from webserver.tasks import task_keeper
from webserver.utilities.configuration import configuration

logger = logging.getLogger(__name__)

logger.info("prott5_annotations imported")

featureExtractor = None
la = None
be = None

if "prott5_annotations" in configuration['celery']['celery_worker_type']:
    from bio_embeddings.extract.basic import BasicAnnotationExtractor
    from bio_embeddings.extract.light_attention import LightAttentionAnnotationExtractor
    from bio_embeddings.extract.bindEmbed21 import BindEmbed21DLAnnotationExtractor

    from bio_embeddings.utilities import convert_list_of_enum_to_string

    logger.info("Loading the feature extraction models...")

    featureExtractor = BasicAnnotationExtractor(
        "t5_xl_u50_from_publication",
        secondary_structure_checkpoint_file=configuration['prottrans_t5_xl_u50']['secondary_structure_checkpoint_file'],
        subcellular_location_checkpoint_file=configuration['prottrans_t5_xl_u50']['subcellular_location_checkpoint_file']
    )

    la = LightAttentionAnnotationExtractor(None, 
        membrane_checkpoint_file=configuration['prottrans_t5_xl_u50']['la_solubility_checkpoint_file'],
        subcellular_location_checkpoint_file=configuration['prottrans_t5_xl_u50']['la_subcellular_location_checkpoint_file']
    )

    be = BindEmbed21DLAnnotationExtractor(
        model_1_file=configuration['bindembed21']['model_1_file'],
        model_2_file=configuration['bindembed21']['model_2_file'],
        model_3_file=configuration['bindembed21']['model_3_file'],
        model_4_file=configuration['bindembed21']['model_4_file'],
        model_5_file=configuration['bindembed21']['model_5_file']
    )

    logger.info("Finished initializing.")


@task_keeper.task()
def get_prott5_annotations_sync(embedding: List) -> Dict[str, str]:
    embedding = np.asarray(embedding)

    annotations = featureExtractor.get_annotations(embedding)
    la_annotations = la.get_subcellular_location(embedding)
    be_annotations = be.get_binding_residues(embedding)

    return {
        "predictedBindingMetal": convert_list_of_enum_to_string(be_annotations.metal_ion),
        "predictedBindingNucleicAcids": convert_list_of_enum_to_string(be_annotations.nucleic_acids),
        "predictedBindingSmallMolecules": convert_list_of_enum_to_string(be_annotations.small_molecules),
        "predictedMembrane": la_annotations.membrane.value,
        "predictedSubcellularLocalizations": la_annotations.localization.value,
        "predictedDSSP3": convert_list_of_enum_to_string(annotations.DSSP3),
        "predictedDSSP8": convert_list_of_enum_to_string(annotations.DSSP8),
        "predictedDisorder": convert_list_of_enum_to_string(annotations.disorder),
        "predictedCCO": [],
        "predictedBPO": [],
        "predictedMFO": [],
        "meta": {
            "predictedBindingMetal": "bindEmbed21, https://www.nature.com/articles/s41598-021-03431-4",
            "predictedBindingNucleicAcids": "bindEmbed21, https://www.nature.com/articles/s41598-021-03431-4",
            "predictedBindingSmallMolecules": "bindEmbed21, https://www.nature.com/articles/s41598-021-03431-4",
            "predictedDSSP3": "ProtT5Sec, https://arxiv.org/pdf/2007.06225",
            "predictedDSSP8": "ProtT5Sec, https://arxiv.org/pdf/2007.06225",
            "predictedDisorder": "ProtT5Sec, https://arxiv.org/pdf/2007.06225",
            "predictedCCO": "unavailable",
            "predictedBPO": "unavailable",
            "predictedMFO": "unavailable",
            "predictedMembrane": "LA_ProtT5, https://www.biorxiv.org/content/10.1101/2021.04.25.441334v1",
            "predictedSubcellularLocalizations": "LA_ProtT5, https://www.biorxiv.org/content/10.1101/2021.04.25.441334v1"
        }
    }

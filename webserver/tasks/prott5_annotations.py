import logging
import numpy as np

from typing import Dict, List

from webserver.tasks import task_keeper
from webserver.utilities.configuration import configuration

logger = logging.getLogger(__name__)

logger.info("prott5_annotations imported")

featureExtractor = None
reference_embeddings = None
reference_identifiers = None
BPO_annotations = None
CCO_annotations = None
MFO_annotations = None
metric = "euclidean"
k = 25
la = None
be = None
tmbed = None

if "prott5_annotations" in configuration['celery']['celery_worker_type']:
    from bio_embeddings.extract.basic import BasicAnnotationExtractor
    from bio_embeddings.extract.light_attention import LightAttentionAnnotationExtractor
    from bio_embeddings.extract.bindEmbed21 import BindEmbed21DLAnnotationExtractor
    from bio_embeddings.extract.tmbed import TmbedAnnotationExtractor

    from bio_embeddings.utilities import convert_list_of_enum_to_string

    logger.info("Loading the feature extraction models...")

    featureExtractor = BasicAnnotationExtractor(
        "t5_xl_u50_from_publication",
        secondary_structure_checkpoint_file=configuration['prottrans_t5_xl_u50']['secondary_structure_checkpoint_file'],
        subcellular_location_checkpoint_file=configuration['prottrans_t5_xl_u50']['subcellular_location_checkpoint_file']
    )


    la = LightAttentionAnnotationExtractor(
        "la_prott5",
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

    tmbed = TmbedAnnotationExtractor(
        model_0_file=configuration['tmbed']['model_0_file'],
        model_1_file=configuration['tmbed']['model_1_file'],
        model_2_file=configuration['tmbed']['model_2_file'],
        model_3_file=configuration['tmbed']['model_3_file'],
        model_4_file=configuration['tmbed']['model_4_file']
    )
    
    logger.info("Loading goa reference embeddings and annotations...")
    import h5py
    from pandas import read_csv, DataFrame
    from sklearn.metrics import pairwise_distances
    from bio_embeddings.extract import get_k_nearest_neighbours

    reference_embeddings = list()

    with h5py.File(configuration["goa"]["go_reference_embeddings"], "r") as embeddings_file:
        reference_identifiers = list(embeddings_file.keys())

        for refereince_identifier in reference_identifiers:
            reference_embeddings.append(np.array(embeddings_file[refereince_identifier]))

    BPO_annotations = read_csv(configuration["goa"]["bpo"], sep=";")
    CCO_annotations = read_csv(configuration["goa"]["cco"], sep=";")
    MFO_annotations = read_csv(configuration["goa"]["mfo"], sep=";")

    logger.info("Finished initializing.")


@task_keeper.task()
def get_prott5_annotations_sync(embedding: List) -> Dict[str, str]:
    embedding = np.asarray(embedding)

    annotations = featureExtractor.get_annotations(embedding)
    la_annotations = la.get_subcellular_location(embedding)
    be_annotations = be.get_binding_residues(embedding)

    # TMbed needs special things: Batch dimension as first dim + length of sequences
    tmbed_annotations = tmbed.get_membrane_residues(embedding[None, ], [len(embedding)])

    # GOA
    reduced_embedding = np.array(embedding).mean(0)

    distance_matrix = pairwise_distances(
        [reduced_embedding],
        reference_embeddings,
        metric=metric,
        n_jobs=-1
    )

    k_nn_indices, k_nn_distances = get_k_nearest_neighbours(distance_matrix, k)
    k_nn_identifiers = list(map(reference_identifiers.__getitem__, k_nn_indices[0]))

    # GoPredSim scales distances/similarities to a reliability index.
    # Note that the following was only asserted for metric='euclidean' or 'cosine'
    k_nn_RI = [0.5 / (0.5 + dist) for dist in k_nn_distances[0]]

    k_nns = DataFrame({metric: k_nn_distances[0], "RI": k_nn_RI}, index=k_nn_identifiers)

    k_nn_BPO = BPO_annotations \
        .join(k_nns, on="identifier") \
        .dropna() \
        .sort_values("RI", ascending=False) \
        .drop_duplicates(subset=["GO_Term"], keep="first")
    k_nn_CCO = CCO_annotations \
        .join(k_nns, on="identifier") \
        .dropna() \
        .sort_values("RI", ascending=False) \
        .drop_duplicates(subset=["GO_Term"], keep="first")
    k_nn_MFO = MFO_annotations \
        .join(k_nns, on="identifier") \
        .dropna() \
        .sort_values("RI", ascending=False) \
        .drop_duplicates(subset=["GO_Term"], keep="first")

    return {
        "predictedBindingMetal": convert_list_of_enum_to_string(be_annotations.metal_ion),
        "predictedBindingNucleicAcids": convert_list_of_enum_to_string(be_annotations.nucleic_acids),
        "predictedBindingSmallMolecules": convert_list_of_enum_to_string(be_annotations.small_molecules),
        "predictedMembrane": la_annotations.membrane.value,
        "predictedSubcellularLocalizations": la_annotations.localization.value,
        "predictedDSSP3": convert_list_of_enum_to_string(annotations.DSSP3),
        "predictedDSSP8": convert_list_of_enum_to_string(annotations.DSSP8),
        "predictedDisorder": convert_list_of_enum_to_string(annotations.disorder),
        "predictedTransmembrane": convert_list_of_enum_to_string(tmbed_annotations[0].membrane_residues),
        "predictedCCO": k_nn_CCO.to_dict("records"),
        "predictedBPO": k_nn_BPO.to_dict("records"),
        "predictedMFO": k_nn_MFO.to_dict("records"),
        "meta": {
            "predictedBindingMetal": "bindEmbed21, https://www.nature.com/articles/s41598-021-03431-4",
            "predictedBindingNucleicAcids": "bindEmbed21, https://www.nature.com/articles/s41598-021-03431-4",
            "predictedBindingSmallMolecules": "bindEmbed21, https://www.nature.com/articles/s41598-021-03431-4",
            "predictedDSSP3": "ProtT5Sec, https://arxiv.org/pdf/2007.06225",
            "predictedDSSP8": "ProtT5Sec, https://arxiv.org/pdf/2007.06225",
            "predictedDisorder": "ProtT5Sec, https://arxiv.org/pdf/2007.06225",
            "predictedCCO": "goPredSim ProtT5, https://github.com/Rostlab/goPredSim#performance-assessment",
            "predictedBPO": "goPredSim ProtT5, https://github.com/Rostlab/goPredSim#performance-assessment",
            "predictedMFO": "goPredSim ProtT5, https://github.com/Rostlab/goPredSim#performance-assessment",
            "predictedMembrane": "LA_ProtT5, https://www.biorxiv.org/content/10.1101/2021.04.25.441334v1",
            "predictedSubcellularLocalizations": "LA_ProtT5, https://www.biorxiv.org/content/10.1101/2021.04.25.441334v1",
            "predictedTransmembrane": "TMbed, https://doi.org/10.1101/2022.06.12.495804"
        }
    }

import logging
from typing import Dict, List

import numpy as np

from webserver.tasks import task_keeper
from webserver.utilities.configuration import configuration

logger = logging.getLogger(__name__)

featureExtractor = None
reference_embeddings = None
reference_identifiers = None
BPO_annotations = None
CCO_annotations = None
MFO_annotations = None
metric = "euclidean"
k = 25

if "seqvec_annotations" in configuration['celery']['celery_worker_type']:
    from bio_embeddings.extract.basic import BasicAnnotationExtractor
    from bio_embeddings.utilities import convert_list_of_enum_to_string

    logger.info("Loading the feature extraction models...")

    featureExtractor = BasicAnnotationExtractor(
        "seqvec_from_publication",
        secondary_structure_checkpoint_file=configuration['seqvec']['secondary_structure_checkpoint_file'],
        subcellular_location_checkpoint_file=configuration['seqvec']['subcellular_location_checkpoint_file']
    )

    logger.info("Loading goa reference embeddings and annotations...")
    import h5py
    from pandas import read_csv, DataFrame
    from sklearn.metrics import pairwise_distances
    from bio_embeddings.extract import get_k_nearest_neighbours

    reference_embeddings = list()

    with h5py.File(configuration["seqvec"]["go_reference_embeddings"], "r") as embeddings_file:
        reference_identifiers = list(embeddings_file.keys())

        for refereince_identifier in reference_identifiers:
            reference_embeddings.append(np.array(embeddings_file[refereince_identifier]))

    BPO_annotations = read_csv(configuration["goa"]["bpo"], sep=";")
    CCO_annotations = read_csv(configuration["goa"]["cco"], sep=";")
    MFO_annotations = read_csv(configuration["goa"]["mfo"], sep=";")

    logger.info("Finished initializing.")


@task_keeper.task()
def get_seqvec_annotations_sync(embedding: List) -> Dict[str, str]:
    embedding = np.asarray(embedding)

    annotations = featureExtractor.get_annotations(embedding)

    l1_embedding = np.array(embedding[1]).mean(0)

    distance_matrix = pairwise_distances(
        [l1_embedding],
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
    k_nn_BPO = BPO_annotations\
        .join(k_nns, on="identifier")\
        .dropna()\
        .sort_values("RI", ascending=False)\
        .drop_duplicates(subset=["GO_Term"], keep="first")
    k_nn_CCO = CCO_annotations\
        .join(k_nns, on="identifier")\
        .dropna()\
        .sort_values("RI", ascending=False)\
        .drop_duplicates(subset=["GO_Term"], keep="first")
    k_nn_MFO = MFO_annotations\
        .join(k_nns, on="identifier")\
        .dropna()\
        .sort_values("RI", ascending=False)\
        .drop_duplicates(subset=["GO_Term"], keep="first")

    return {
        "predictedDSSP3": convert_list_of_enum_to_string(annotations.DSSP3),
        "predictedDSSP8": convert_list_of_enum_to_string(annotations.DSSP8),
        "predictedDisorder": convert_list_of_enum_to_string(annotations.disorder),
        "predictedMembrane": annotations.membrane.value,
        "predictedSubcellularLocalizations": annotations.localization.value,
        "predictedCCO": k_nn_CCO.to_dict("records"),
        "predictedBPO": k_nn_BPO.to_dict("records"),
        "predictedMFO": k_nn_MFO.to_dict("records"),
        "meta": {
            "predictedDSSP3": "SeqVecSec, https://doi.org/10.1186/s12859-019-3220-8",
            "predictedDSSP8": "SeqVecSec, https://doi.org/10.1186/s12859-019-3220-8",
            "predictedDisorder": "SeqVecSec, https://doi.org/10.1186/s12859-019-3220-8",
            "predictedCCO": "goPredSim SeqVec, https://doi.org/10.1038/s41598-020-80786-0",
            "predictedBPO": "goPredSim SeqVec, https://doi.org/10.1038/s41598-020-80786-0",
            "predictedMFO": "goPredSim SeqVec, https://doi.org/10.1038/s41598-020-80786-0",
            "predictedMembrane": "SeqVecLoc, https://doi.org/10.1186/s12859-019-3220-8",
            "predictedSubcellularLocalizations": "SeqVecLoc, https://doi.org/10.1186/s12859-019-3220-8",
        }
    }

import io
from datetime import datetime
from typing import Dict, Tuple

import numpy
import numpy as np
from werkzeug.exceptions import abort

from webserver.database import get_embedding_cache, get_features_cache, get_residue_landscape_cache
# Prott5
from webserver.tasks.prott5_embeddings import get_prott5_embeddings_sync
from webserver.tasks.prott5_annotations import get_prott5_annotations_sync
from webserver.tasks.prott5_residue_landscape_annotations import get_residue_landscape_output_sync


def get_embedding(model_name: str, sequence: str) -> np.array:
    """
    Notes regarding the caching: tobytes is really only the raw array and
    doesn't contain dtype and shape, so we're saving the shape separately
    and know that the dtype is numpy.float64
    """
    cached = get_embedding_cache.find_one(
        {"model_name": model_name, "sequence": sequence}
    )
    if cached:
        return numpy.frombuffer(cached["array"], dtype=numpy.float64).reshape(
            cached["shape"]
        )

    model = {
        "prottrans_t5_xl_u50": get_prott5_embeddings_sync
    }.get(model_name)

    if not model:
        return abort(400, f"Model '{model_name}' isn't available.")

    # time_limit && soft_time_limit limit the execution time. Expires limits the queuing time.
    job = model.apply_async(
        args=[sequence], time_limit=60 * 5, soft_time_limit=60 * 5, expires=60 * 60
    )
    array = np.array(job.get())
    assert array.dtype == numpy.float64, array.dtype

    if len(sequence) < 500:
        get_embedding_cache.insert_one(
            {
                "uploadDate": datetime.utcnow(),
                "model_name": model_name,
                "sequence": sequence,
                "shape": array.shape,
                "array": array.tobytes(),
            }
        )
    return array


def get_features(model_name: str, sequence: str) -> Dict[str, str]:
    """
    Calls two jobs:
    - First job gets the embeddings (can be run on GPU machine with little system RAM)
    - Second job gets the features (can be run on CPU -- if GoPredSim integrated, might be >2GB RAM)

    Original implementation run one job, but this might be wasteful of good resources and limits execution to host with
    > 4GB RAM!
    """
    cached = get_features_cache.find_one(
        {"model_name": model_name, "sequence": sequence}
    )
    if cached:
        return cached["features"]

    embeddings = get_embedding(model_name, sequence)

    annotation_model = {
        "prottrans_t5_xl_u50": get_prott5_annotations_sync,
    }.get(model_name)

    job = annotation_model.apply_async(
        args=[embeddings.tolist()],
        time_limit=60 * 5,
        soft_time_limit=60 * 5,
        expires=60 * 60,
    )

    features = job.get()

    get_features_cache.insert_one(
        {
            "uploadDate": datetime.utcnow(),
            "model_name": model_name,
            "sequence": sequence,
            "features": features,
        }
    )
    return features


def get_residue_landscape(model_name: str, sequence: str) -> dict:
    cached = get_residue_landscape_cache.find_one(
        {"model_name": model_name, "sequence": sequence}
    )
    if cached:
        predicted_conservation_values = numpy.frombuffer(cached['conservation_prediction'], dtype=numpy.int8).reshape(
            cached['conservation_prediction_shape']).tolist()
        predicted_variation_values = numpy.frombuffer(cached['variation_prediction'], dtype=numpy.int8).reshape(
            cached['variation_prediction_shape']).tolist()

        predictedVariation = {'x_axis': cached['x_axis'], 'y_axis': cached['y_axis'],
                              'values': predicted_variation_values}

        predictedConservation = {'x_axis': cached['x_axis'], 'y_axis': cached['y_axis'],
                                 'values': predicted_conservation_values}

        predictedClasses = numpy.frombuffer(cached["classes_prediction"], dtype=np.int8).tolist()

        meta_data = cached['meta']

        return {'predictedConservation': predictedConservation,
                'predictedVariation': predictedVariation,
                'predictedClasses': predictedClasses,
                'meta': meta_data}

    embedding_as_list = get_embedding(model_name, sequence).tolist()

    residue_landscape_model = {
        "prottrans_t5_xl_u50": get_residue_landscape_output_sync,
    }.get(model_name)

    job = residue_landscape_model.apply_async(
        args=[sequence, embedding_as_list], time_limit=60 * 5, soft_time_limit=60 * 5, expires=60 * 60
    )

    residue_landscape_worker_out = job.get()

    meta_data = residue_landscape_worker_out['meta']

    predictedVariation = residue_landscape_worker_out['predictedVariation']

    predictedConservation = residue_landscape_worker_out['predictedConservation']

    predictedClasses = residue_landscape_worker_out['predictedClasses']

    predictedClasses_arr = np.array(predictedClasses, dtype=np.int8)

    predicted_conservation_values = np.array(predictedVariation['values'], dtype=np.int8)

    predicted_variation_values = np.array(predictedVariation['values'], dtype=np.int8)

    assert predictedVariation['x_axis'] == predictedConservation['x_axis']
    assert predictedVariation['y_axis'] == predictedConservation['y_axis']

    get_residue_landscape_cache.insert_one(
        {
            "uploadDate": datetime.utcnow(),
            "model_name": "prottrans_t5_xl_u50",
            "sequence": sequence,
            "conservation_prediction_shape": predicted_conservation_values.shape,
            "conservation_prediction": predicted_conservation_values.tobytes(),
            "variation_prediction": predicted_variation_values.tobytes(),
            "variation_prediction_shape": predicted_variation_values.shape,
            "classes_prediction": predictedClasses_arr.tobytes(),
            "x_axis": predictedVariation['x_axis'],
            "y_axis": predictedVariation['y_axis'],
            "meta": meta_data
        }
    )

    return {'predictedConservation': predictedConservation,
            'predictedVariation': predictedVariation,
            'predictedClasses': predictedClasses,
            'meta': meta_data}

from datetime import datetime
from time import sleep
from typing import Dict

import numpy
import numpy as np
from werkzeug.exceptions import abort

from webserver.database import get_embedding_cache, get_features_cache, get_structure_cache, get_structure_jobs, \
    JOB_PENDING, JOB_DONE

# Prott5
from webserver.tasks.prott5_embeddings import get_prott5_embeddings_sync
from webserver.tasks.prott5_annotations import get_prott5_annotations_sync
# Colabfold
from webserver.tasks.structure import get_structure_colabfold


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


def _insert_structure_to_db(sequence: str, structure: Dict[str, object]) -> Dict[str, object]:
    result = {
        'uploadDate': datetime.utcnow(),
        'sequence': sequence,
        'structure': structure,
    }
    get_structure_cache.insert_one(result)
    return result['structure']


def get_structure(sequence: str) -> Dict[str, object]:
    "".join(sequence.split())
    sequence = sequence.upper()
    cached = get_structure_cache.find_one(
        {'sequence': sequence}
    )
    if cached:
        return cached['structure']

    in_progress = get_structure_jobs.find_one(
        {'sequence': sequence}
    )
    if in_progress:
        if in_progress['status'] == JOB_PENDING:
            sleep(2.0)
            return get_structure(sequence)
        elif in_progress['status'] == JOB_DONE:
            return get_structure(sequence)

    job = get_structure_colabfold.apply_async(
        args=[sequence],
        time_limit=60 * 15,
        soft_time_limit=60 * 15,
        expires=60 * 60,
    )
    result_dict = job.get()

    # If the worker detects that there is a pending or finished job, he only returns the status of the job in the DB
    if 'status' in result_dict:
        if result_dict['status'] == JOB_PENDING:
            sleep(2.0)
        return get_structure(sequence)
    else:
        try:
            ret = _insert_structure_to_db(sequence, job.get())
        # We can not mark the job as done before the results are in the database.
        # If there is any exception while inserting the structure to the database, we must mark the job as failed
        except Exception as e:
            get_structure_jobs.delete_one(
                {
                    'sequence': sequence,
                    'status': JOB_PENDING,
                }
            )
            raise e
        else:
            get_structure_jobs.find_one({
                'sequence': sequence,
                'status': JOB_PENDING,
            }

            )
            get_structure_jobs.update_one(
                {
                    'sequence': sequence,
                    'status': JOB_PENDING,
                },
                {
                    '$set':
                        {
                            'timestamp': datetime.utcnow(),
                            'status': JOB_DONE,
                        }
                }
            )
            return ret


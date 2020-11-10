from functools import lru_cache
from typing import Dict

import numpy as np
from werkzeug.exceptions import abort

from webserver.tasks.protbert_embeddings import get_protbert_embeddings_sync
from webserver.tasks.protbert_annotations import get_protbert_annotations_sync
from webserver.tasks.seqvec_embeddings import get_seqvec_embeddings_sync
from webserver.tasks.seqvec_annotations import get_seqvec_annotations_sync


@lru_cache()
def get_embedding(model_name: str, sequence: str) -> np.array:
    model = {
        'seqvec': get_seqvec_embeddings_sync,
        'prottrans_bert_bfd': get_protbert_embeddings_sync
    }.get(model_name)

    if not model:
        return abort(400, f"Model '{model_name}' isn't available.")

    # time_limit && soft_time_limit limit the execution time. Expires limits the queuing time.
    job = model.apply_async(args=[sequence], time_limit=60 * 5, soft_time_limit=60 * 5, expires=60 * 60)
    return np.array(job.get())


@lru_cache()
def get_feaures(model_name: str, sequence: str) -> Dict[str, str]:
    """
    Calls two jobs:
    - First job gets the emebddings (can be run on GPU machine with little system RAM)
    - Second job gets the features (can be run on CPU -- if GoPredSim integrated, might be >2GB RAM)

    Original implementation run one job, but this might be wasteful of good resources and limits execution to host with
    > 4GB RAM!
    """
    embeddings = get_embedding(model_name, sequence)

    annotation_model = {
        'seqvec': get_seqvec_annotations_sync,
        'prottrans_bert_bfd': get_protbert_annotations_sync
    }.get(model_name)

    job = annotation_model.apply_async(args=[embeddings.tolist()], time_limit=60 * 5, soft_time_limit=60 * 5, expires=60 * 60)
    return job.get()

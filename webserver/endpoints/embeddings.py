import io
import h5py
import numpy as np

from functools import lru_cache

from flask import request, send_file, abort
from flask_restx import Resource

from webserver.endpoints import api
from webserver.endpoints.request_models import sequence_post_parameters
from webserver.endpoints.utils import check_valid_sequence
from webserver.tasks.seqvec_embeddings import get_seqvec_embeddings_sync
from webserver.tasks.protbert_embeddings import get_protbert_embeddings_sync

ns = api.namespace("embeddings", description="Calculate embeddings on the fly.")


@lru_cache()
def _get_embedding(model_name: str, sequence: str) -> np.array:
    model = {
        'seqvec': get_seqvec_embeddings_sync,
        'prottrans_bert_bfd': get_protbert_embeddings_sync
    }.get(model_name)

    if not model:
        return abort(400, f"Model '{model_name}' isn't available.")

    # time_limit && soft_time_limit limit the execution time. Expires limits the queuing time.
    job = model.apply_async(args=[sequence], time_limit=60 * 5, soft_time_limit=60 * 5, expires=60 * 60)
    return np.array(job.get())


@ns.route('')
class Embeddings(Resource):
    @api.expect(sequence_post_parameters, validate=True)
    @api.response(200, "Returns an hdf5 file with one dataset called `sequence` "
                       "containing the embedding of the supplied sequence.")
    @api.response(400, "Invalid input. See return message for details.")
    @api.response(505, "Server error")
    def post(self):
        params = request.json

        sequence = params.get('sequence')

        if not sequence or len(sequence) > 2000 or not check_valid_sequence(sequence):
            return abort(400, "Sequence is too long or contains invalid characters.")

        model_name = params.get('model', 'seqvec')

        embedding = _get_embedding(model_name, sequence)

        buffer = io.BytesIO()
        with h5py.File(buffer, "w") as embeddings_file:
            embeddings_file.create_dataset("sequence", data=embedding)

        # This simulates closing the file and re-opening it.
        # Otherwise the cursor will already be at the end of the
        # file when flask tries to read the contents, and it will
        # think the file is empty.
        buffer.seek(0)

        return send_file(buffer, attachment_filename="embeddings_file.h5", as_attachment=True)

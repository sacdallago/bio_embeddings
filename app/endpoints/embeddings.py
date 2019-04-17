import io
import numpy as np
from flask import request, abort, send_file
from flask_restplus import Resource
from app.endpoints import api
from app.endpoints.utils import check_valid_sequence
from app.machine_learning import get_seqvec
from app.endpoints.request_models import sequence_post_parameters
from app.tasks import task_keeper

ns = api.namespace("embeddings", description="Get embeddings")


@ns.route('')
class Embeddings(Resource):
    @api.expect(sequence_post_parameters, validate=True)
    @api.response(200, "Calculated embeddings")
    @api.response(400, "Invalid input. Most likely the sequence is too long, or contains invalid characters.")
    @api.response(505, "Server error")
    def post(self):
        params = request.json

        sequence = params.get('sequence')

        if not sequence or len(sequence) > 2000 or not check_valid_sequence(sequence):
            return abort(400)

        job = get_seqvec.delay(sequence)
        embeddings = job.get()

        buffer = io.BytesIO()

        np.savez_compressed(buffer, embeddings)

        buffer.seek(0)  # This simulates closing the file and re-opening it.
        #  Otherwise the cursor will already be at the end of the
        #  file when flask tries to read the contents, and it will
        #  think the file is empty.

        return send_file(buffer, attachment_filename="embedding.npy")

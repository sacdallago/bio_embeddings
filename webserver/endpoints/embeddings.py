import io
import h5py

from flask import request, send_file, abort
from flask_restx import Resource

from webserver.endpoints import api
from webserver.endpoints.request_models import sequence_post_parameters
from webserver.endpoints.task_interface import get_embedding
from webserver.endpoints.utils import check_valid_sequence

ns = api.namespace("embeddings", description="Calculate embeddings on the fly.")


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

        model_name = params.get('model', 'prottrans_t5_xl_u50')

        embedding = get_embedding(model_name, sequence)

        buffer = io.BytesIO()
        with h5py.File(buffer, "w") as embeddings_file:
            embeddings_file.create_dataset("sequence", data=embedding)

        # This simulates closing the file and re-opening it.
        # Otherwise the cursor will already be at the end of the
        # file when flask tries to read the contents, and it will
        # think the file is empty.
        buffer.seek(0)

        return send_file(buffer, attachment_filename="embeddings_file.h5", as_attachment=True)

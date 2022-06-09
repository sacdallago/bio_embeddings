import io
import h5py

from flask import request, send_file, abort
from flask_restx import Resource

from webserver.endpoints import api
from webserver.endpoints.request_models import vespa_post_parameters
from webserver.endpoints.task_interface import get_residue_landscape
from webserver.endpoints.task_interface import get_embedding
from webserver.endpoints.utils import check_valid_sequence

ns = api.namespace("VESPA", description="Calculate SAV effect")


@ns.route('')
class vespa(Resource):
    @api.expect(vespa_post_parameters, validate=True)
    @api.response(200, "Returns an hdf5 file with one dataset called `sequence` "
                       "containing the embedding_buffer of the supplied sequence.")
    @api.response(400, "Invalid input. See return message for details.")
    @api.response(505, "Server error")
    def post(self):
        params = request.json

        sequence = params.get('sequence')

        if not sequence or len(sequence) > 2000 or not check_valid_sequence(sequence):
            return abort(400, "Sequence is too long or contains invalid characters.")

        model_name = 'prottrans_t5_xl_u50'

        embedding = get_embedding(model_name, sequence)
        embedding = embedding.tolist()

        residue_landscape_output = get_residue_landscape(sequence=sequence,model_name='prottrans_t5_xl_u50', embedding_as_list=embedding)
        cons_pred = residue_landscape_output['predictedConservation']
        vespa_out = residue_landscape_output['predictedVariation']

        return {'predictedVariation':vespa_out,
                'predictedConservation':cons_pred}

import io
import h5py

from flask import request, send_file, abort
from flask_restx import Resource

from webserver.endpoints import api
from webserver.endpoints.annotations import ns
from webserver.endpoints.request_models import residue_landscape_post_parameters
from webserver.endpoints.task_interface import get_residue_landscape
from webserver.endpoints.utils import check_valid_sequence

#ns = api.namespace("Residue_landscape", description="Compute conservation and variation")


@ns.route('/residue/landscape')
class residue_landscape(Resource):
    @api.expect(residue_landscape_post_parameters, validate=True)
    @api.response(200, "Returns an hdf5 file with one dataset called `sequence` "
                       "containing the embedding_buffer of the supplied sequence.")
    @api.response(400, "Invalid input. See return message for details.")
    @api.response(505, "Server error")
    def post(self):
        params = request.json

        sequence = params.get('sequence')

        if not sequence or len(sequence) > 2000 or not check_valid_sequence(sequence):
            return abort(400, "Sequence is too long or contains invalid characters.")

        residue_landscape_output = get_residue_landscape(model_name='prottrans_t5_xl_u50',sequence=sequence)
        cons_pred = residue_landscape_output['predictedConservation']
        variation = residue_landscape_output['predictedVariation']

        return {'predictedVariation': variation,
                'predictedConservation': cons_pred}

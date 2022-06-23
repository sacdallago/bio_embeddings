from flask import request
from flask_restx import Resource

from webserver.endpoints import api
from webserver.endpoints.request_models import sequence_get_parameters_structure, sequence_post_parameters_structure
from webserver.endpoints.task_interface import get_structure
from webserver.endpoints.utils import abort, check_valid_sequence

ns = api.namespace("structure", description="Get structure predictions on the fly.")


def _get_structure_from_params(params):
    predictor = params.get('predictor', "colabfold")
    sequence = params.get('sequence')
    if not sequence or len(sequence) > 500 or not check_valid_sequence(sequence):
        return abort(400, "Sequence is too long or contains invalid characters.")
    if predictor not in {"colabfold"}:
        return abort(400, "Invalid predictor specified")

    ret = get_structure(predictor, sequence)
    return ret


@ns.route('')
class Annotations(Resource):
    @api.expect(sequence_get_parameters_structure, validate=True)
    @api.response(200, "Annotations in specified format")
    @api.response(201, "Created job for structure prediction")
    @api.response(202, "Structure prediction request accepted")
    @api.response(400, "Invalid input. See return message for details.")
    @api.response(505, "Server error")
    def get(self):
        return _get_structure_from_params(request.args)

    @api.expect(sequence_post_parameters_structure, validate=True)
    @api.response(200, "Annotations in specified format")
    @api.response(201, "Created job for structure prediction")
    @api.response(202, "Structure prediction request accepted")
    @api.response(400, "Invalid input. See return message for details.")
    @api.response(505, "Server error")
    def post(self):
        return _get_structure_from_params(request.json)

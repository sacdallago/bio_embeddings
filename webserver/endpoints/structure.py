from flask import request
from flask_restx import Resource

from webserver.endpoints import api
from webserver.endpoints.request_models import sequence_get_parameters_structure, sequence_post_parameters_structure
from webserver.endpoints.task_interface import get_structure

ns = api.namespace("structure", description="Get structure predictions on the fly.")


@ns.route('')
class Annotations(Resource):
    @api.expect(sequence_get_parameters_structure, validate=True)
    @api.response(200, "Annotations in specified format")
    @api.response(400, "Invalid input. See return message for details.")
    @api.response(505, "Server error")
    def get(self):
        params = request.args
        return get_structure(params.get('predictor', 'colabfold'), params.get('sequence'))

    @api.expect(sequence_post_parameters_structure, validate=True)
    @api.response(200, "Annotations in specified format")
    @api.response(400, "Invalid input. See return message for details.")
    @api.response(505, "Server error")
    def post(self):
        params = request.json
        return get_structure(params.get('predictor', 'colabfold'), params.get('sequence'))

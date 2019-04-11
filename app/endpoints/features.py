from app.endpoints import api
from flask import request, abort
from app.machine_learning import get_seqvec, get_subcellular_location, get_secondary_structure
from flask_restplus import Resource
from app.endpoints.request_models import sequence_post_parameters

ns = api.namespace("features", description="Get features from embeddings through sequence")


@ns.route('')
class Features(Resource):
    @api.expect(sequence_post_parameters, validate=True)
    @api.response(200, "Got features embeddings")
    @api.response(400, "Invalid input. Most likely the sequence is too long, or contains invalid characters.")
    @api.response(505, "Server error")
    def post(self):
        params = request.json

        sequence = params.get('sequence')

        if not sequence:
            return abort(400)
        if len(sequence) > 2000:
            return abort(400)

        embeddings = get_seqvec(sequence)

        predicted_localizations, predicted_membrane = get_subcellular_location(embeddings)
        predicted_dssp3, predicted_dssp8, predicted_disorder = get_secondary_structure(embeddings)

        return predicted_dssp8

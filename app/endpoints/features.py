from flask import request, abort, jsonify
from flask_restplus import Resource
from app.endpoints import api
from app.endpoints.utils import check_valid_sequence
from app.endpoints.request_models import sequence_post_parameters
from app.machine_learning import get_seqvec, get_subcellular_location, get_secondary_structure

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

        if not sequence or len(sequence) > 2000 or not check_valid_sequence(sequence):
            return abort(400)

        job = get_seqvec.delay(sequence)
        embeddings = job.get()

        predicted_localizations, predicted_membrane = get_subcellular_location(embeddings)
        predicted_dssp3, predicted_dssp8, predicted_disorder = get_secondary_structure(embeddings)

        result = {
            "sequence": sequence,
            "predictedSubcellularLocalizations": predicted_localizations,
            "predictedMembrane": predicted_membrane,
            "predictedDSSP3": predicted_dssp3,
            "predictedDSSP8": predicted_dssp8,
            "predictedDisorder": predicted_disorder,
        }

        return jsonify(result)

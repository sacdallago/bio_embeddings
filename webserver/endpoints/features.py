from flask import request, abort, jsonify
from flask_restplus import Resource
from webserver.endpoints import api
from webserver.endpoints.utils import check_valid_sequence
from webserver.endpoints.request_models import sequence_post_parameters
from webserver.machine_learning import get_seqvec, get_subcellular_location, get_secondary_structure

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

        # time_limit && soft_time_limit limit the execution time. Expires limits the queuing time.
        job = get_seqvec.apply_async(args=[sequence], time_limit=60*5, soft_time_limit=60*5, expires=60*60)
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

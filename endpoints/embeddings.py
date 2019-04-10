from endpoints import api
from flask_cors import cross_origin
from flask import request, abort

from machine_learning.extractor import get_seqvec
from flask_restplus import Resource
from endpoints.request_models import sequence_post_parameters


ns = api.namespace("embeddings", description="Get embeddings")


@ns.route('/')
@cross_origin()
class Embeddings(Resource):
    @api.expect(sequence_post_parameters, validate=True)
    @api.response(200, "Calculated embeddings")
    @api.response(400, "Invalid input. Most likely the sequence is too long, or contains invalid characters.")
    @api.response(505, "Server error")
    def post(self):
        params = request.json

        sequence = params.get('sequence')

        if not sequence :
            return abort(400)
        if len(sequence) > 2000 :
            return abort(400)

        embeddings = get_seqvec(sequence)

        return embeddings

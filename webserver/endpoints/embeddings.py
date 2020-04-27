from flask import request, jsonify
from flask_restplus import Resource
from webserver.endpoints import api
from webserver.endpoints.utils import validate_file_submission
# from webserver.tasks.embeddings import get_embeddings
from webserver.endpoints.request_models import file_post_parser

ns = api.namespace("embeddings", description="Get embeddings")


@ns.route('')
class Embeddings(Resource):
    @api.expect(file_post_parser, validate=True)
    @api.response(200, "Calculated embeddings")
    @api.response(400, "Invalid input. Most likely the sequence is too long, or contains invalid characters.")
    @api.response(505, "Server error")
    def post(self):
        pass


@ns.route('/validate')
class Embeddings(Resource):
    @api.expect(file_post_parser, validate=True)
    @api.response(200, "Validated sequence and annotations")
    @api.response(400, "Invalid input. Either the submitted files are not in the correct format, not submitted or don't contain all required fields.")
    @api.response(505, "Server error")
    def post(self):
        file_validation = validate_file_submission(request)

        statistics = file_validation['statistics']

        return jsonify(statistics)

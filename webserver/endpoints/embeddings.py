import uuid
from flask import request, jsonify, send_file, abort
from flask_restplus import Resource
from tempfile import NamedTemporaryFile
from bio_embeddings.utilities import write_fasta_file
from webserver.database import write_file, get_file
from webserver.endpoints import api
from webserver.endpoints.utils import validate_file_submission
from webserver.tasks.embeddings import get_embeddings
from webserver.endpoints.request_models import file_post_parser, request_status_parser, request_results_parser

ns = api.namespace("embeddings", description="Get embeddings")


@ns.route('')
class Embeddings(Resource):
    @api.expect(file_post_parser, validate=True)
    @api.response(200, "Calculated embeddings")
    @api.response(400, "Invalid input. Most likely the sequence is too long, or contains invalid characters.")
    @api.response(505, "Server error")
    def post(self):
        file_validation = validate_file_submission(request)

        job_id = uuid.uuid4().hex

        temp_file = NamedTemporaryFile()
        write_fasta_file(file_validation['sequences'], temp_file.name)
        write_file(job_id, "sequences_file", temp_file.name)

        temp_file = NamedTemporaryFile()
        file_validation['annotations'].to_csv(temp_file.name)
        write_file(job_id, "annotations_file", temp_file.name)

        async_call = get_embeddings.apply_async(args=(job_id, 'seqvec'), task_id=job_id)

        return {'request_id': async_call.id, 'job_id': job_id}

    @api.expect(request_results_parser, validate=True)
    @api.response(200, "Found embedding file: downloading")
    @api.response(505, "Server error")
    def get(self):
        job_id = request.args.get('id')
        file_request = request.args.get('file', 'reduced_embeddings_file')

        if get_embeddings.AsyncResult(job_id).status == "SUCCESS":
            file = get_file(job_id, file_request)
            if file:
                return send_file(file, attachment_filename=file_request, as_attachment=True)
            else:
                return abort(404, "File {} not found".format(file_request))
        else:
            abort(404, "Job not found or not completed.")


@ns.route('/status')
class Embeddings(Resource):
    @api.expect(request_status_parser, validate=True)
    @api.response(200, "Returns job status in celery queue.")
    @api.response(505, "Server error")
    def get(self):
        job_id = request.args.get('id')

        return {"status": get_embeddings.AsyncResult(job_id).status}


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

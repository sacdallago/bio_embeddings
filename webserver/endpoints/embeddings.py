import uuid
from flask import request, send_file, abort
from flask_restx import Resource
from webserver.database import get_file
from webserver.endpoints import api
from webserver.endpoints.utils import validate_FASTA_submission
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
        statistics, sequences = validate_FASTA_submission(request)
        job_id = uuid.uuid4().hex
        pipeline_type = request.form.get('pipeline_type', 'annotations_from_bert')

        async_call = get_embeddings.apply_async(args=(job_id, sequences, pipeline_type),
                                                task_id=job_id)

        return {'request_id': async_call.id, 'job_id': job_id}

    @api.expect(request_results_parser, validate=True)
    @api.response(200, "Found embedding file: downloading")
    @api.response(505, "Server error")
    def get(self):
        job_id = request.args.get('id')
        file_request = request.args.get('file', 'reduced_embeddings_file')

        job_status = get_embeddings.AsyncResult(job_id).status

        if job_status == "SUCCESS" or job_status == "STARTED":
            file = get_file(job_id, file_request)
            if file:
                return send_file(file, attachment_filename="reduced_embeddings_file.h5", as_attachment=True)
            else:
                return abort(404, f"File {file_request} not found."
                                  f"Either requesting invalid file or job not finished yet."
                                  f"Job status {job_status}.")
        else:
            abort(404, "Job not found or not completed.")


@ns.route('/status')
class Embeddings(Resource):
    @api.expect(request_status_parser, validate=True)
    @api.response(200, "Returns job status in celery queue.")
    @api.response(505, "Server error")
    def get(self):
        job_id = request.args.get('id')

        # TODO: add a list of available files for the job (aka mongo search by job id)
        return {"status": get_embeddings.AsyncResult(job_id).status}

import uuid

from flask import request, send_file, abort
from flask_restx import Resource

from webserver.database import get_file, get_list_of_files
from webserver.endpoints import api
from webserver.endpoints.request_models import (
    file_post_parser,
    request_status_parser,
    request_results_parser,
)
from webserver.endpoints.utils import validate_FASTA_submission
from webserver.tasks.pipeline import run_pipeline

ns = api.namespace("pipeline", description="Run pipeline jobs")


@ns.route('')
class Pipeline(Resource):
    @api.expect(file_post_parser, validate=True)
    @api.response(200, "Calculated embeddings")
    @api.response(400, "Invalid input. See message for details")
    @api.response(505, "Server error")
    def post(self):
        validated_request = validate_FASTA_submission(request)
        job_id = uuid.uuid4().hex
        pipeline_type = request.form.get('pipeline_type', 'annotations_from_bert')

        async_call = run_pipeline.apply_async(args=(job_id, validated_request.sequences, pipeline_type),
                                              task_id=job_id)

        return {'request_id': async_call.id, 'job_id': job_id}

    @api.expect(request_status_parser, validate=True)
    @api.response(200, "Returns job status in celery queue.")
    @api.response(505, "Server error")
    def get(self):
        job_id = request.args.get('id')
        job_status = run_pipeline.AsyncResult(job_id).status

        return {
            "status": job_status,
            "files": get_list_of_files(job_id)
        }


_extensions = {
    "embeddings_file": ".h5",
    "reduced_embeddings_file": ".h5",
    "sequence_file": ".fasta",
    "DSSP3_predictions_file": ".fasta",
    "DSSP8_predictions_file": ".fasta",
    "disorder_predictions_file": ".fasta",
    "per_sequence_predictions_file": '.csv',
    "mapping_file": ".csv",
    "plot_file": ".html"
}


@ns.route('/download')
class PipelineDownload(Resource):
    @api.expect(request_results_parser, validate=True)
    @api.response(200, "Found embedding file: downloading")
    @api.response(505, "Server error")
    def get(self):
        job_id = request.args.get('id')
        file_request = request.args.get('file', 'embeddings_file')

        job_status = run_pipeline.AsyncResult(job_id).status

        file = get_file(job_id, file_request)
        if file:
            return send_file(
                file,
                attachment_filename=file_request+_extensions.get(file_request, ""),
                as_attachment=True)
        else:
            return abort(
                404,
                f"File {file_request} not found. "
                f"Either requesting invalid file or job not finished yet. "
                f"Job status {job_status}.",
            )

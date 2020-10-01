from webserver.endpoints import api
from werkzeug.datastructures import FileStorage

file_post_parser = api.parser()
file_post_parser.add_argument('sequences', location='files', type=FileStorage, required=True, help='UTF8 binary with FASTA sequences.')
file_post_parser.add_argument('pipeline_type', location='form', type=str, required=False, help='What pipeline to run.')

request_status_parser = api.parser()
request_status_parser.add_argument('id', location='args', type=str, required=True, help='Job id.')

request_results_parser = api.parser()
request_results_parser.add_argument('id', location='args', type=str, required=True, help='Job id.')
request_results_parser.add_argument('file', location='args', type=str, required=False, help='Name of the file to be dowloaded.')

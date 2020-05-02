from webserver.endpoints import api
from werkzeug.datastructures import FileStorage

file_post_parser = api.parser()
file_post_parser.add_argument('sequences', location='files', type=FileStorage, required=True)

request_status_parser = api.parser()
request_status_parser.add_argument('id', location='args', help='Job id', required=True)

request_results_parser = api.parser()
request_results_parser.add_argument('id', location='args', help='Job id', required=True)
request_results_parser.add_argument('file', location='args', help='Which file to download', required=False)

from webserver.endpoints import api
from werkzeug.datastructures import FileStorage

file_post_parser = api.parser()
file_post_parser.add_argument('sequences', location='files', type=FileStorage, required=True)
file_post_parser.add_argument('annotations', location='files', type=FileStorage, required=True)

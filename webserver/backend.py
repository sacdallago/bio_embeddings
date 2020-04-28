import os
import yaml
from pathlib import Path
from flask import Flask, Blueprint
from flask_cors import CORS
from webserver.endpoints import api
from webserver.endpoints.embeddings import ns as embeddings_namespace
from webserver.endpoints.visualize import create_dash_app

# Read and load configuration file
configuration = dict()
module_dir = Path(os.path.dirname(os.path.abspath(__file__)))

with open(module_dir / "backend_configuration.yml", 'r') as stream:
    try:
        configuration = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Initialize API

app = Flask(__name__)
dash_app = create_dash_app(app)

# Required parameters
app.config['MAX_CONTENT_LENGTH'] = eval(configuration['max_content_length'])

blueprint = Blueprint('api', __name__)
cors = CORS(blueprint, origins=configuration.get('origins', []))

api.init_app(blueprint)
api.add_namespace(embeddings_namespace)
app.register_blueprint(blueprint)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

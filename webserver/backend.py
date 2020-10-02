from flask import Flask, Blueprint, render_template
from flask_cors import CORS
from webserver.endpoints import api
from webserver.endpoints.embeddings import ns as embeddings_namespace
from webserver.utilities.configuration import configuration

# Initialize API
app = Flask(__name__)

# Required parameters
app.config['MAX_CONTENT_LENGTH'] = configuration['web']['max_content_length']

blueprint = Blueprint('api', __name__, url_prefix='/api')
cors = CORS(blueprint, origins=['https://embed.protein.properties', 'http://localhost:3000'])

api.init_app(blueprint)
api.add_namespace(embeddings_namespace)
app.register_blueprint(blueprint)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

from flask import Flask, Blueprint
from flask_cors import CORS

from webserver.endpoints import api
from webserver.endpoints.embeddings import ns as embeddings_namespace
from webserver.endpoints.features import ns as features_namespace

app = Flask(__name__)

blueprint = Blueprint('api', __name__)
cors = CORS(blueprint, origins=['http://localhost:3000', 'https://embed.protein.properties'])

api.init_app(blueprint)
api.add_namespace(embeddings_namespace)
api.add_namespace(features_namespace)

app.register_blueprint(blueprint)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

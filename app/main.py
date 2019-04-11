from flask import Flask, Blueprint

from app.endpoints import api
from app.endpoints.embeddings import ns as embeddings_namespace
from app.endpoints.features import ns as features_namespace

app = Flask(__name__)

blueprint = Blueprint('api', __name__)
api.init_app(blueprint)
api.add_namespace(embeddings_namespace)
api.add_namespace(features_namespace)

app.register_blueprint(blueprint)

if __name__ == '__main__':
    app.run()

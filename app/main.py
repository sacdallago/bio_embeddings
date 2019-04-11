from flask import Flask, Blueprint

from app.endpoints import api
from app.endpoints.embeddings import ns as embeddings_namespace


app = Flask(__name__)

blueprint = Blueprint('api', __name__)
api.init_app(blueprint)
api.add_namespace(embeddings_namespace)

app.register_blueprint(blueprint)

if __name__ == '__main__':
    app.run()

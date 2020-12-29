from flask import Flask, Blueprint, render_template
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

from webserver.endpoints import api
from webserver.endpoints.pipeline import ns as pipeline_namespace
from webserver.endpoints.embeddings import ns as embeddings_namespace
from webserver.endpoints.annotations import ns as annotations_namespace

from webserver.utilities.configuration import configuration

from webserver.tasks import task_keeper

# Initialize API
app = Flask(__name__)

# Required parameters
app.config['MAX_CONTENT_LENGTH'] = configuration['web']['max_content_length']

blueprint = Blueprint('api', __name__, url_prefix='/api')
cors = CORS(blueprint, origins=['https://embed.protein.properties', 'http://localhost:3000',
                                'https://predictprotein.org', 'https://predictprotein.test',
                                'https://bioembeddings.com', 'https://seqvec.bioembeddings.com',
                                ])

api.init_app(blueprint)
api.add_namespace(pipeline_namespace)
api.add_namespace(embeddings_namespace)
api.add_namespace(annotations_namespace)

app.register_blueprint(blueprint)

# https://flask.palletsprojects.com/en/1.1.x/quickstart/#hooking-in-wsgi-middleware
app.wsgi_app = ProxyFix(app.wsgi_app)


@app.route("/")
def index():
    i = task_keeper.control.inspect()
    queues = set()
    active_queues = i.active_queues()
    for queue in active_queues:
        queues.add(active_queues[queue][0]['name'])
    return render_template("index.html", workers=queues)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

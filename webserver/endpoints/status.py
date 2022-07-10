from flask import abort
from flask_restx import Resource
from webserver.endpoints import api
from webserver.endpoints.utils import get_queues

ns = api.namespace("status", description="Get the status of the workers.")

_workers = [
    'prott5',
    'prott5_annotations',
    'pipeline',
    'colabfold',
    'prott5_residue_landscape_annotations'
]


@ns.route('')
class Status(Resource):
    @api.response(200, "Returns an object with active workers.")
    @api.response(505, "Server error")
    def get(self):
        queues = get_queues()

        active_workers = dict()

        for worker in _workers:
            active_workers[worker] = worker in queues

        return active_workers


@ns.route('/<worker>')
class Status(Resource):
    @api.response(200, "Queue for worker exists. Use /status to get a list of workers.")
    @api.response(503, "Queue for worker does not exist")
    @api.response(505, "Server error")
    def get(self, worker=""):
        queues = get_queues()

        if worker in queues:
            return "OK"
        else:
            return abort(503, "Worker is unavailable.")

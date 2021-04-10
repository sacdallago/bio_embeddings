from flask import abort
from flask_restx import Resource
from webserver.endpoints import api
from webserver.tasks import task_keeper


ns = api.namespace("status", description="Get the status of the workers.")

_workers = [
    'seqvec',
    'seqvec_annotations',
    'protbert',
    'protbert_annotations',
    'prott5',
    'prott5_annotations',
    'pipeline',
]


def _get_queues():
    i = task_keeper.control.inspect()
    queues = set()
    active_queues = i.active_queues() or dict()
    for queue in active_queues:
        queues.add(active_queues[queue][0]["name"])

    return queues


@ns.route('')
class Status(Resource):
    @api.response(200, "Returns an object with active workers.")
    @api.response(505, "Server error")
    def get(self):
        queues = _get_queues()

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
        queues = _get_queues()

        if worker in queues:
            return "OK"
        else:
            return abort(503, "Worker is unavailable.")

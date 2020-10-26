from flask import request, abort
from flask_restx import Resource

from webserver.utilities.parsers import (
    Source, Evidence, annotations_to_protvista_converter, SecondaryStructure, Disorder
)
from webserver.endpoints import api
from webserver.endpoints.request_models import sequence_post_parameters_annotations
from webserver.endpoints.utils import check_valid_sequence
from webserver.tasks.seqvec_embeddings import get_seqvec_annotations_sync
from webserver.tasks.protbert_embeddings import get_protbert_annotations_sync

ns = api.namespace("annotations", description="Get annotations on the fly.")


@ns.route('')
class Annotations(Resource):
    @api.expect(sequence_post_parameters_annotations, validate=True)
    @api.response(200, "Embedding in npy format")
    @api.response(400, "Invalid input. See return message for details.")
    @api.response(505, "Server error")
    def post(self):
        params = request.json

        sequence = params.get('sequence')

        if not sequence or len(sequence) > 2000 or not check_valid_sequence(sequence):
            return abort(400, "Sequence is too long or contains invalid characters.")

        model = {
            'seqvec': get_seqvec_annotations_sync,
            'prottrans_bert_bfd': get_protbert_annotations_sync
        }.get(params.get('model', 'seqvec'))

        if not model:
            return abort(400, f"Model '{params.get('model')}' isn't available.")

        # time_limit && soft_time_limit limit the execution time. Expires limits the queuing time.
        job = model.apply_async(args=[sequence], time_limit=60*5, soft_time_limit=60*5, expires=60*60)
        annotations = job.get()

        annotations['sequence'] = sequence

        format = params.get('format', 'legacy')

        if format == "protvista-predictprotein":
            source = Source(
                url=request.url,
                id=sequence,
                name="bio_embeddings"
            )

            evidence = Evidence(
                source=source,
            )

            protvista_features = dict()
            protvista_features['sequence'] = sequence

            protvista_features['features'] = list()
            protvista_features['features'].extend(
                annotations_to_protvista_converter(
                    features_string=annotations['predictedDSSP8'],
                    evidences=[evidence],
                    type="SECONDARY_STRUCTURE_8_STATES_(SEQVEC)",
                    feature_enum=SecondaryStructure
                )
            )
            protvista_features['features'].extend(
                annotations_to_protvista_converter(
                    features_string=annotations['predictedDSSP3'],
                    evidences=[evidence],
                    type="SECONDARY_STRUCTURE_3_STATES_(SEQVEC)",
                    feature_enum=SecondaryStructure
                )
            )
            protvista_features['features'].extend(
                annotations_to_protvista_converter(
                    features_string=annotations['predictedDisorder'],
                    evidences=[evidence],
                    type="DISORDER_(SEQVEC)",
                    feature_enum=SecondaryStructure
                )
            )

            return protvista_features
        else:
            return annotations

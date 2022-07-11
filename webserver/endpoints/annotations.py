from itertools import groupby
from flask import request, abort
from flask_restx import Resource

from webserver.endpoints import api
from webserver.endpoints.request_models import sequence_post_parameters_annotations, sequence_get_parameters_annotations, \
    residue_landscape_post_parameters
from webserver.endpoints.task_interface import get_features
from webserver.endpoints.task_interface import get_residue_landscape
from webserver.endpoints.utils import check_valid_sequence, get_queues
from webserver.utilities.parsers import (
    Source, Evidence, annotations_to_protvista_converter,
    SecondaryStructure, Disorder, BindingResidues, MembraneResidues
)

ns = api.namespace("annotations", description="Get annotations on the fly.")


def _filter_ontology(annotations, ontology):
    if annotations.get(ontology):
        first_k = next(goannotations for identifier, goannotations in groupby(annotations[ontology], lambda x: x["identifier"]))
        annotations[ontology] = list(first_k)


def _get_annotations_from_params(params):
    sequence = params.get('sequence')

    if not sequence or len(sequence) > 2000 or not check_valid_sequence(sequence):
        return abort(400, "Sequence is too long or contains invalid characters.")

    model_name = params.get('model', 'prottrans_t5_xl_u50')
    annotations = get_features(model_name, sequence)

    if model_name == 'prottrans_t5_xl_u50' and 'prott5_residue_landscape_annotations' in get_queues():
        residue_landscape_output = get_residue_landscape(model_name=model_name, sequence=sequence)
        # merge the output of the residue landscape into the feature dict
        # add the meta information
        for key in residue_landscape_output['meta']:
            annotations['meta'][key] = residue_landscape_output['meta'][key]

        residue_landscape_output.pop('meta', None)

        # add all the remaining information
        for key in residue_landscape_output:
            annotations[key] = residue_landscape_output[key]

    annotations['sequence'] = sequence

    format = params.get('format', 'legacy')
    only_closest_k = params.get('only_closest_k', True)

    if only_closest_k == True:
        _filter_ontology(annotations, "predictedBPO")
        _filter_ontology(annotations, "predictedCCO")
        _filter_ontology(annotations, "predictedMFO")

    if format == "protvista-predictprotein":
        source = Source(
            url=request.url,
            id="sync",
            name=f"bio_embeddings using {model_name}"
        )

        evidence = Evidence(
            source=source,
        )

        protvista_features = dict()
        protvista_features['sequence'] = sequence

        protvista_features['features'] = list()
        if annotations.get('predictedDSSP8'):
            protvista_features['features'].extend(
                annotations_to_protvista_converter(
                    features_string=annotations['predictedDSSP8'],
                    evidences=[evidence],
                    type=f"SECONDARY_STRUCTURE_8_STATES_({model_name})", feature_enum=SecondaryStructure
                )
            )
        if annotations.get('predictedDSSP3'):
            protvista_features['features'].extend(
                annotations_to_protvista_converter(
                    features_string=annotations['predictedDSSP3'],
                    evidences=[evidence],
                    type=f"SECONDARY_STRUCTURE_3_STATES_({model_name})",
                    feature_enum=SecondaryStructure
                )
            )
        if annotations.get('predictedDisorder'):
            protvista_features['features'].extend(
                annotations_to_protvista_converter(
                    features_string=annotations['predictedDisorder'],
                    evidences=[evidence],
                    type=f"DISORDER_({model_name})",
                    feature_enum=Disorder
                )
            )
        if annotations.get('predictedBindingMetal'):
            protvista_features['features'].extend(
                annotations_to_protvista_converter(
                    features_string=annotations['predictedBindingMetal'],
                    evidences=[evidence],
                    type=f"BINDING_METAL_({model_name})",
                    feature_enum=BindingResidues
                )
            )
        if annotations.get('predictedBindingNucleicAcids'):
            protvista_features['features'].extend(
                annotations_to_protvista_converter(
                    features_string=annotations['predictedBindingNucleicAcids'],
                    evidences=[evidence],
                    type=f"BINDING_NUCLEIC_ACIDS_({model_name})",
                    feature_enum=BindingResidues
                )
            )
        if annotations.get('predictedBindingSmallMolecules'):
            protvista_features['features'].extend(
                annotations_to_protvista_converter(
                    features_string=annotations['predictedBindingSmallMolecules'],
                    evidences=[evidence],
                    type=f"BINDING_SMALL_MOLECULES_({model_name})",
                    feature_enum=BindingResidues
                )
            )
        if annotations.get('predictedTransmembrane'):
            protvista_features['features'].extend(
                annotations_to_protvista_converter(
                    features_string=annotations['predictedTransmembrane'],
                    evidences=[evidence],
                    type=f"TRANSMEMBRANE_({model_name})",
                    feature_enum=MembraneResidues
                )
            )

        return protvista_features
    elif format == "legacy":
        predictedCCO = {}
        predictedBPO = {}
        predictedMFO = {}

        for prediction in annotations['predictedCCO']:
            predictedCCO[prediction['GO_Term']] = max(predictedCCO.get(prediction['GO_Term'], -1), prediction['RI'])

        for prediction in annotations['predictedBPO']:
            predictedBPO[prediction['GO_Term']] = max(predictedBPO.get(prediction['GO_Term'], -1), prediction['RI'])

        for prediction in annotations['predictedMFO']:
            predictedMFO[prediction['GO_Term']] = max(predictedMFO.get(prediction['GO_Term'], -1), prediction['RI'])

        annotations['predictedCCO'] = predictedCCO
        annotations['predictedBPO'] = predictedBPO
        annotations['predictedMFO'] = predictedMFO

        return annotations
    elif format == "go-predictprotein":
        mapping_function = lambda x: {
            "gotermid": x['GO_Term'],
            "gotermname": x['GO_Name'],
            "gotermscore": round(x['RI'] * 100)
        }

        predictedCCO = {
            "ontology": "Cellular Component Ontology",
            "goTermWithScore": list(map(mapping_function, annotations['predictedCCO']))
        }
        predictedBPO = {
            "ontology": "Biological Process Ontology",
            "goTermWithScore": list(map(mapping_function, annotations['predictedBPO']))
        }
        predictedMFO = {
            "ontology": "Molecular Function Ontology",
            "goTermWithScore": list(map(mapping_function, annotations['predictedMFO']))
        }

        return [predictedBPO, predictedCCO, predictedMFO]
    elif format == "full":
        return annotations
    else:
        abort(400, f"Wrong format passed: {format}")


@ns.route('')
class Annotations(Resource):
    @api.expect(sequence_get_parameters_annotations, validate=True)
    @api.response(200, "Annotations in specified format")
    @api.response(400, "Invalid input. See return message for details.")
    @api.response(505, "Server error")
    def get(self):
        params = request.args

        return _get_annotations_from_params(params)

    @api.expect(sequence_post_parameters_annotations, validate=True)
    @api.response(200, "Annotations in specified format")
    @api.response(400, "Invalid input. See return message for details.")
    @api.response(505, "Server error")
    def post(self):
        params = request.json

        return _get_annotations_from_params(params)


@ns.route('/residue/landscape')
class residue_landscape(Resource):
    @api.expect(residue_landscape_post_parameters, validate=True)
    @api.response(200, "Returns an hdf5 file with one dataset called `sequence` "
                       "containing the embedding_buffer of the supplied sequence.")
    @api.response(400, "Invalid input. See return message for details.")
    @api.response(505, "Server error")
    def post(self):
        params = request.json

        sequence = params.get('sequence')

        if not sequence or len(sequence) > 2000 or not check_valid_sequence(sequence):
            return abort(400, "Sequence is too long or contains invalid characters.")

        return get_residue_landscape(model_name='prottrans_t5_xl_u50', sequence=sequence)

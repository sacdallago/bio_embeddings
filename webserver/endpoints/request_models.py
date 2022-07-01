from webserver.endpoints import api
from werkzeug.datastructures import FileStorage
from flask_restx import fields

file_post_parser = api.parser()
file_post_parser.add_argument(
    'sequences',
    location='files',
    type=FileStorage,
    required=True,
    help='UTF8 binary with FASTA sequences.'
)
file_post_parser.add_argument(
    'pipeline_type',
    location='form',
    type=str,
    required=False,
    help='What pipeline to run.'
)

request_status_parser = api.parser()
request_status_parser.add_argument(
    'id',
    location='args',
    type=str,
    required=True,
    help='Job id.'
)

request_results_parser = api.parser()
request_results_parser.add_argument(
    'id',
    location='args',
    type=str,
    required=True,
    help='Job id.'
)
request_results_parser.add_argument(
    'file',
    location='args',
    type=str,
    required=False,
    help='Name of the file to be dowloaded.'
)

sequence_post_parameters = api.parser()
sequence_post_parameters.add_argument(
    'sequence',
    location='json',
    type=str,
    required=True,
    help='Protein sequence in AA format.'
)

lm_field = fields.String(
    location='json',
    description='Which LM to use; options: prottrans_t5_xl_u50.',
    required=False,
    default='prottrans_t5_xl_u50',
    example='prottrans_t5_xl_u50'
)

sequence_field = fields.String(
    location='json',
    description='Protein sequence in AA format.',
    required=True,
    example='MALLHSARVLSGVASAFHPGLAAAASARASSWWAHVEMGPPDPILGVTEAYKRDTNSKKMNLGVGAYRDDNGKPYVLPSVRKAEAQIAAKGLDKEYLPIGGLAEFCRASAELALGENSEVVKSGRFVTVQTISGTGALRIGASFLQRFFKFSRDVFLPKPSWGNHTPIFRDAGMQLQSYRYYDPKTCGFDFTGALEDISKIPEQSVLLLHACAHNPTGVDPRPEQWKEIATVVKKRNLFAFFDMAYQGFASGDGDKDAWAVRHFIEQGINVCLCQSYAKNMGLYGERVGAFTVICKDADEAKRVESQLKILIRPMYSNPPIHGARIASTILTSPDLRKQWLQEVKGMADRIIGMRTQLVSNLKKEGSTHSWQHITDQIGMFCFTGLKPEQVERLTKEFSIYMTKDGRISVAGVTSGNVGYLAHAIHQVTK'
)

predictor_field = fields.String(
    location='json',
    description='Which structure predictor to use; options: colabfold.',
    required=False,
    default='colabfold',
    example='colabfold'
)

sequence_post_parameters = api.model('sequence_post', {
    'model': lm_field,
    'sequence': sequence_field
})

fromat_field = fields.String(
    location='json',
    description='Output format. Options: legacy (default), protvista-predictprotein, go-predictprotein, full',
    required=False,
    default='legacy',
    example='protvista-predictprotein'
)

k_neighbours_field = fields.Boolean(
    location='json',
    description='Boolean to filter GoPredSim for closest hit only. '
                'Default set to True, if False, considers up to 3 nearest-neighbours.',
    required=False,
    default=True
)

sequence_post_parameters_annotations = api.model('sequence_post_annotations', {
    'model': lm_field,
    'sequence': sequence_field,
    'format': fromat_field,
    'only_closest_k': k_neighbours_field
})

sequence_get_parameters_annotations = api.parser()
sequence_get_parameters_annotations.add_argument(
    'sequence',
    location='args',
    type=str,
    required=True,
    help='Protein sequence in AA format.'
)
sequence_get_parameters_annotations.add_argument(
    'model',
    location='args',
    type=str,
    required=False,
    help='Which LM to use; options: prottrans_t5_xl_u50.'
)
sequence_get_parameters_annotations.add_argument(
    'format',
    location='args',
    type=str,
    required=False,
    help='Output format. Options: legacy (default), protvista-predictprotein, go-predictprotein, full'
)

residue_landscape_post_parameters = api.model('residue_landscape_post_sequence', {
    'sequence': sequence_field
})

sequence_get_parameters_structure = api.parser()
sequence_get_parameters_structure.add_argument(
    'sequence',
    location='args',
    type=str,
    required=True,
    help='Protein sequence in AA format.'
)
sequence_get_parameters_structure.add_argument(
    'predictor',
    location='args',
    type=str,
    required=False,
    help='Which structure predictor to use; options: colabfold'
)

sequence_post_parameters_structure = api.model('sequence_post_structure', {
    'predictor': predictor_field,
    'sequence': sequence_field,
})

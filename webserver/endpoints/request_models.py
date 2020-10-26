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

sequence_post_parameters = api.model('sequence_post', {
    'model': fields.String(
        location='json',
        description='Which LM to use; options: seqvec, prottrans_bert_bfd.',
        required=False,
        default='seqvec',
        example='seqvec'
    ),
    'sequence': fields.String(
            location='json',
            description='Protein sequence in AA format.',
            required=True,
            example='MALLHSARVLSGVASAFHPGLAAAASARASSWWAHVEMGPPDPILGVTEAYKRDTNSKKMNLGVGAYRDDNGKPYVLPSVRKAEAQIAAKGLDKEYLPIGGLAEFCRASAELALGENSEVVKSGRFVTVQTISGTGALRIGASFLQRFFKFSRDVFLPKPSWGNHTPIFRDAGMQLQSYRYYDPKTCGFDFTGALEDISKIPEQSVLLLHACAHNPTGVDPRPEQWKEIATVVKKRNLFAFFDMAYQGFASGDGDKDAWAVRHFIEQGINVCLCQSYAKNMGLYGERVGAFTVICKDADEAKRVESQLKILIRPMYSNPPIHGARIASTILTSPDLRKQWLQEVKGMADRIIGMRTQLVSNLKKEGSTHSWQHITDQIGMFCFTGLKPEQVERLTKEFSIYMTKDGRISVAGVTSGNVGYLAHAIHQVTK'
        ),
})

import h5py

from os import path
from io import StringIO
from pathlib import Path
from Bio import SeqIO, SeqRecord
from typing import Iterable, Dict
from tempfile import TemporaryDirectory

from bio_embeddings.utilities import write_fasta_file
from bio_embeddings.utilities.pipeline import execute_pipeline_from_config
from bio_embeddings.utilities.config import read_config_file


from webserver.database import get_file, write_file
from webserver.tasks import task_keeper
from webserver.utilities.configuration import configuration

_module_dir: Path = Path(path.dirname(path.abspath(__file__)))

# Template configs: these will be the "job types" available
_annotations_from_bert: Dict[str, Dict[str, str]] = read_config_file(_module_dir / "template_configs" / 'annotations_from_bert.yml')
_annotations_from_seqvec: Dict[str, Dict[str, str]] = read_config_file(_module_dir / "template_configs" / 'annotations_from_seqvec.yml')

# Enrich templates with execution specific parameters: location of weights & optionable max_aa

# BERT
_annotations_from_bert['bert_embeddings']['model_directory'] = configuration['bert']['model_directory']
_annotations_from_bert['bert_embeddings']['max_amino_acids'] = configuration['bert']['max_amino_acids']
_annotations_from_bert['annotations_from_bert']['secondary_structure_checkpoint_file'] = configuration['bert']['secondary_structure_checkpoint_file']
_annotations_from_bert['annotations_from_bert']['subcellular_location_checkpoint_file'] = configuration['bert']['subcellular_location_checkpoint_file']


# SEQVEC
_annotations_from_seqvec['seqvec_embeddings']['weights_file'] = configuration['seqvec']['weights_file']
_annotations_from_seqvec['seqvec_embeddings']['options_file'] = configuration['seqvec']['options_file']
_annotations_from_seqvec['seqvec_embeddings']['max_amino_acids'] = configuration['seqvec']['max_amino_acids']
_annotations_from_seqvec['annotations_from_seqvec']['secondary_structure_checkpoint_file'] = configuration['seqvec']['secondary_structure_checkpoint_file']
_annotations_from_seqvec['annotations_from_seqvec']['subcellular_location_checkpoint_file'] = configuration['seqvec']['subcellular_location_checkpoint_file']


_CONFIGS = {
    'annotations_from_seqvec': _annotations_from_seqvec,
    'annotations_from_bert': _annotations_from_bert,
}

_FILES_TO_STORE = [
        "embeddings_file"
        "reduced_embeddings_file",
        "sequence_file",
        "annotations_file",
        "mapping_file"
    ]


@task_keeper.task()
def get_embeddings(job_identifier, sequences, pipeline_type):
    config = _CONFIGS[pipeline_type]

    def _post_stage_save(stage_out_config):
        for file_name in _FILES_TO_STORE:
            if stage_out_config.get(file_name):
                write_file(job_identifier, file_name, stage_out_config[file_name])

    with TemporaryDirectory() as workdir:
        write_fasta_file(sequences, Path(workdir) / "sequences.fasta")

        # Add last job details
        config['global']['prefix'] = Path(workdir) / "bio_embeddings_job"
        config['global']['sequences_file'] = Path(workdir) / "sequences.fasta"

        execute_pipeline_from_config(config, post_stage=_post_stage_save)

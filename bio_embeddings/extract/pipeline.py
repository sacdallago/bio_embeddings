import logging
import h5py
import numpy as np
from copy import deepcopy

from Bio.Seq import Seq
from pandas import read_csv
from typing import Dict, Any
from bio_embeddings.extract.basic import BasicAnnotationExtractor
from bio_embeddings.utilities.remote_file_retriever import get_model_file
from bio_embeddings.utilities.filemanagers import get_file_manager
from bio_embeddings.utilities.helpers import check_required, read_fasta, convert_list_of_enum_to_string, \
    write_fasta_file
from bio_embeddings.utilities.exceptions import InvalidParameterError

logger = logging.getLogger(__name__)


def unsupervised(**kwargs) -> Dict[str, Any]:
    raise NotImplementedError()

    # TODO: pick up from here
    check_required(kwargs, ['reference_embeddings', 'reference_annotations'])

    result_kwargs = deepcopy(kwargs)

    return result_kwargs


def seqvec_from_publication(**kwargs) -> Dict[str, Any]:
    return predict_annotations_using_basic_models("seqvec_from_publication", **kwargs)


def bert_from_publication(**kwargs) -> Dict[str, Any]:
    return predict_annotations_using_basic_models("bert_from_publication", **kwargs)


def predict_annotations_using_basic_models(model, **kwargs) -> Dict[str, Any]:
    """
    Protocol extracts secondary structure (DSSP3 and DSSP8), disorder, subcellular location and membrane boundness
    from "embeddings_file". Embeddings can either be generated with SeqVec or ProtBert.
    SeqVec models are used in this publication: https://doi.org/10.1186/s12859-019-3220-8
    ProtTrans models are used in this publication: https://doi.org/10.1101/2020.07.12.199554

    :param model: either "bert_from_publication" or "seqvec_from_publication". Used to download files
    """

    check_required(kwargs, ['embeddings_file', 'mapping_file', 'remapped_sequences_file'])
    necessary_files = ['secondary_structure_checkpoint_file', 'subcellular_location_checkpoint_file']
    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager(**kwargs)

    # Download necessary files if needed
    for file in necessary_files:
        if not result_kwargs.get(file):
            file_path = file_manager.create_file(result_kwargs.get('prefix'), result_kwargs.get('stage_name'), file)
            get_model_file(path=file_path, model=f'{model}_annotations_extractors', file=file)
            result_kwargs[file] = file_path

    annotation_extractor = BasicAnnotationExtractor(model, **result_kwargs)

    # mapping file will be needed for protein-wide annotations
    mapping_file = read_csv(result_kwargs['mapping_file'], index_col=0)

    # Try to create final files (if this fails, now is better than later
    DSSP3_predictions_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                           result_kwargs.get('stage_name'),
                                                           'DSSP3_predictions_file',
                                                           extension='.fasta')
    result_kwargs['DSSP3_predictions_file'] = DSSP3_predictions_file_path
    DSSP8_predictions_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                           result_kwargs.get('stage_name'),
                                                           'DSSP8_predictions_file',
                                                           extension='.fasta')
    result_kwargs['DSSP8_predictions_file'] = DSSP8_predictions_file_path
    disorder_predictions_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                              result_kwargs.get('stage_name'),
                                                              'disorder_predictions_file',
                                                              extension='.fasta')
    result_kwargs['disorder_predictions_file'] = disorder_predictions_file_path
    per_sequence_predictions_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                  result_kwargs.get('stage_name'),
                                                                  'per_sequence_predictions_file',
                                                                  extension='.csv')
    result_kwargs['per_sequence_predictions_file'] = per_sequence_predictions_file_path

    # Create sequence containers
    DSSP3_sequences = list()
    DSSP8_sequences = list()
    disorder_sequences = list()

    with h5py.File(result_kwargs['embeddings_file'], 'r') as embedding_file:
        for protein_sequence in read_fasta(result_kwargs['remapped_sequences_file']):

            # Per-AA annotations: DSSP3, DSSP8 and disorder
            embedding = np.array(embedding_file[protein_sequence.id])

            annotations = annotation_extractor.get_annotations(embedding)

            DSSP3_sequence = deepcopy(protein_sequence)
            DSSP3_sequence.seq = Seq(convert_list_of_enum_to_string(annotations.DSSP3))
            DSSP3_sequences.append(DSSP3_sequence)

            DSSP8_sequence = deepcopy(protein_sequence)
            DSSP8_sequence.seq = Seq(convert_list_of_enum_to_string(annotations.DSSP8))
            DSSP8_sequences.append(DSSP8_sequence)

            disorder_sequence = deepcopy(protein_sequence)
            disorder_sequence.seq = Seq(convert_list_of_enum_to_string(annotations.disorder))
            disorder_sequences.append(disorder_sequence)

            # Per-sequence annotations, e.g. subcell loc & membrane boundness
            mapping_file.at[protein_sequence.id, 'subcellular_location'] = annotations.localization.value
            mapping_file.at[protein_sequence.id, 'membrane_or_soluble']  = annotations.membrane.value

    # Write files
    mapping_file.to_csv(per_sequence_predictions_file_path)
    write_fasta_file(DSSP3_sequences, DSSP3_predictions_file_path)
    write_fasta_file(DSSP8_sequences, DSSP8_predictions_file_path)
    write_fasta_file(disorder_sequences, disorder_predictions_file_path)

    return result_kwargs


# list of available extraction protocols
PROTOCOLS = {
    "bert_from_publication": bert_from_publication,
    "seqvec_from_publication": seqvec_from_publication,
    "unsupervised": unsupervised
}


def run(**kwargs):
    """
    Run embedding protocol

    Parameters
    ----------
    kwargs arguments (* denotes optional):
        prefix: Output prefix for all generated files
        protocol: Which protocol to use
        stage_name: The stage name

    Returns
    -------
    Dictionary with results of stage
    """
    check_required(kwargs, ['protocol', 'prefix', 'stage_name'])

    if kwargs["protocol"] not in PROTOCOLS:
        raise InvalidParameterError(
            "Invalid protocol selection: " +
            "{}. Valid protocols are: {}".format(
                kwargs["protocol"], ", ".join(PROTOCOLS.keys())
            )
        )

    return PROTOCOLS[kwargs["protocol"]](**kwargs)

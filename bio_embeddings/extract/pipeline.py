import logging
import h5py
import torch
import numpy as np

from math import isclose
from copy import deepcopy
from typing import Dict, Any, List
from Bio.Seq import Seq
from pandas import read_csv, DataFrame

from bio_embeddings.extract.basic import BasicAnnotationExtractor
from bio_embeddings.utilities.remote_file_retriever import get_model_file
from bio_embeddings.utilities.filemanagers import get_file_manager
from bio_embeddings.utilities.helpers import check_required, read_fasta, convert_list_of_enum_to_string, \
    write_fasta_file
from bio_embeddings.utilities.exceptions import InvalidParameterError

logger = logging.getLogger(__name__)


# From https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/util.py
# and https://github.com/Rostlab/goPredSim/blob/master/two_sample_util.py
def _pairwise_distance_matrix(sample_1: torch.Tensor, sample_2: torch.Tensor, norm: float = 2.0, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    eps :
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    if isclose(norm, 2.0):
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)


def _flatten_2d_list(l: List[List[str]]) -> List[str]:
    return [item for sublist in l for item in sublist]


def unsupervised(**kwargs) -> Dict[str, Any]:
    check_required(kwargs, ['reference_embeddings_file', 'reference_annotations_file', 'reduced_embeddings_file'])

    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager(**kwargs)

    # Try to create final files (if this fails, now is better than later
    transferred_annotations_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                 result_kwargs.get('stage_name'),
                                                                 'transferred_annotations_file',
                                                                 extension='.csv')

    # Read the reference annotations and reference embeddings

    # The reference annotations file must be CSV containing two columns & headers like:
    # identifier,label
    # ** identifier doesn't need to be unique **
    reference_annotations_file = read_csv(result_kwargs['reference_annotations_file'])

    # Save a copy of the annotation file with only necessary cols cols
    input_reference_annotations_file_path = file_manager.create_file(kwargs.get('prefix'),
                                                                     result_kwargs.get('stage_name'),
                                                                     'input_reference_annotations_file',
                                                                     extension='.csv')

    reference_annotations_file.to_csv(input_reference_annotations_file_path, index=False)

    result_kwargs['input_reference_annotations_file'] = input_reference_annotations_file_path

    # Starting from here order is super important!
    reference_identifiers = reference_annotations_file['identifier'].unique()
    reference_identifiers.sort()
    reference_embeddings = list()

    # Save a copy of the reference embeddings file with only necessary embeddings
    input_reference_embeddings_file_path = file_manager.create_file(kwargs.get('prefix'),
                                                                    result_kwargs.get('stage_name'),
                                                                    'input_reference_embeddings_file',
                                                                    extension='.h5')

    result_kwargs['input_reference_embeddings_file'] = input_reference_embeddings_file_path

    # Only read in embeddings for annotated sequences! This will save RAM/GPU_RAM.
    with h5py.File(result_kwargs['reference_embeddings_file'], 'r') as reference_embeddings_file:
        with h5py.File(result_kwargs['input_reference_embeddings_file'], 'w') as input_reference_embeddings_file:
            for identifier in reference_identifiers:
                current_embedding = np.array(reference_embeddings_file[identifier])
                reference_embeddings.append(current_embedding)
                input_reference_embeddings_file.create_dataset(identifier, data=current_embedding)

    # mapping file will be needed to transfer annotations
    mapping_file = read_csv(result_kwargs['mapping_file'], index_col=0)

    # Important to have consistent ordering!
    target_identifiers = mapping_file.index.values
    target_identifiers.sort()
    target_embeddings = list()

    with h5py.File(result_kwargs['reduced_embeddings_file'], 'r') as reduced_embeddings_file:
        for identifier in target_identifiers:
            target_embeddings.append(np.array(reduced_embeddings_file[identifier]))

    # TODO: !!!! IMPORTANT !!!!
    #       KS: for consistency, please make the same changes here, as you are doing on
    #       https://gitlab.lrz.de/sacdallago/bio_embeddings/-/merge_requests/39
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Calculate the pairwise distances and store them as matrix (in CSV format)
    reference_embeddings = torch.tensor(reference_embeddings, device=device).squeeze()
    target_embeddings = torch.tensor(target_embeddings, device=device).squeeze()

    result_kwargs['pairwise_distance_norm'] = result_kwargs.get('pairwise_distance_norm', 2.0)

    pairwise_distances = _pairwise_distance_matrix(
        target_embeddings,
        reference_embeddings,
        norm=result_kwargs['pairwise_distance_norm'])

    pairwise_distances = pairwise_distances.numpy()

    pairwise_distances_matrix_file_path = file_manager.create_file(kwargs.get('prefix'),
                                                                   result_kwargs.get('stage_name'),
                                                                   'pairwise_distances_matrix_file',
                                                                   extension='.csv')
    pairwise_distances_matrix_file = DataFrame(pairwise_distances,
                                               index=target_identifiers,
                                               columns=reference_identifiers)
    pairwise_distances_matrix_file.to_csv(pairwise_distances_matrix_file_path, index=True)
    result_kwargs['pairwise_distances_matrix_file'] = pairwise_distances_matrix_file_path

    # transfer & store annotations
    result_kwargs['k_nearest_neighbours'] = result_kwargs.get('k_nearest_neighbours', 1)

    transferred_annotations = list()

    for index in range(len(target_identifiers)):
        current_annotation = {
            'identifier': target_identifiers[index]
        }

        target_to_reference_distances = pairwise_distances[index, :]
        nearest_neighbour_indices = np.argpartition(
            target_to_reference_distances,
            result_kwargs['k_nearest_neighbours'])[:result_kwargs['k_nearest_neighbours']]

        nearest_neighbour_identifiers = list()
        nearest_neighbour_distances = list()
        nearest_neighbour_annotations = list()

        for i in nearest_neighbour_indices:
            nearest_neighbour_identifiers.append(reference_identifiers[i])
            nearest_neighbour_distances.append(target_to_reference_distances[i])
            reference_annotations_rows = reference_annotations_file[
                reference_annotations_file['identifier'] == reference_identifiers[i]
                ]
            nearest_neighbour_annotations.append(reference_annotations_rows['label'].values)

        current_annotation['transferred_annotations'] = ";".join(list(set(_flatten_2d_list(nearest_neighbour_annotations))))

        for i, (distance, identifier, annotations) in enumerate(
                sorted(
                    list(
                        zip(nearest_neighbour_distances, nearest_neighbour_identifiers, nearest_neighbour_annotations)
                    ),
                    key=lambda x: x[0]
                )):
            current_annotation[f'k_nn_{i+1}_identifier'] = identifier
            current_annotation[f'k_nn_{i+1}_distance'] = distance
            current_annotation[f'k_nn_{i+1}_annotations'] = ";".join(annotations)

        transferred_annotations.append(current_annotation)

    transferred_annotations_dataframe = DataFrame(transferred_annotations)
    transferred_annotations_dataframe = transferred_annotations_dataframe.set_index('identifier')
    transferred_annotations_dataframe = mapping_file.join(transferred_annotations_dataframe)
    transferred_annotations_dataframe.to_csv(transferred_annotations_file_path, index=True)

    result_kwargs['transferred_annotations_file'] = transferred_annotations_file_path

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

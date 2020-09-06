import logging
import h5py
import numpy as np

from copy import deepcopy
from typing import Dict, Any, List
from Bio.Seq import Seq
from pandas import read_csv, DataFrame
from sklearn.metrics import pairwise_distances as _pairwise_distances

from bio_embeddings.extract.basic import BasicAnnotationExtractor
from bio_embeddings.utilities.remote_file_retriever import get_model_file
from bio_embeddings.utilities.filemanagers import get_file_manager
from bio_embeddings.utilities.helpers import check_required, read_fasta, convert_list_of_enum_to_string, \
    write_fasta_file
from bio_embeddings.utilities.exceptions import InvalidParameterError, UnrecognizedEmbeddingError

logger = logging.getLogger(__name__)


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
    input_reference_annotations_file_path = file_manager.create_file(result_kwargs.get('prefix'),
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
    input_reference_embeddings_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                    result_kwargs.get('stage_name'),
                                                                    'input_reference_embeddings_file',
                                                                    extension='.h5')

    result_kwargs['input_reference_embeddings_file'] = input_reference_embeddings_file_path

    # Only read in embeddings for annotated sequences! This will save RAM/GPU_RAM.
    with h5py.File(result_kwargs['reference_embeddings_file'], 'r') as reference_embeddings_file:
        # Sanity check: check that all identifiers in reference_annotation_file are present as embeddings

        unembedded_identifiers = set(reference_identifiers) - set(reference_embeddings_file.keys())

        if len(unembedded_identifiers) > 0:
            raise UnrecognizedEmbeddingError("Your reference_annotations_file includes identifiers for which "
                                             "no embedding can be found in your reference_embeddings_file.\n"
                                             "We require the set of identifiers in the reference_annotations_file "
                                             "to be a equal or a subset of the embeddings present in the "
                                             "reference_embeddings_file.\n"
                                             "To fix this issue, you can use the "
                                             "bio_embeddings.utilities.remove_identifiers_from_annotations_file "
                                             "function (see notebooks). "
                                             "The faulty identifiers are:\n['" +
                                             "','".join(unembedded_identifiers) + "']")

        with h5py.File(result_kwargs['input_reference_embeddings_file'], 'w') as input_reference_embeddings_file:
            for identifier in reference_identifiers:
                current_embedding = np.array(reference_embeddings_file[identifier])
                reference_embeddings.append(current_embedding)
                input_reference_embeddings_file.create_dataset(identifier, data=current_embedding)

    # mapping file will be needed to transfer annotations
    mapping_file = read_csv(result_kwargs['mapping_file'], index_col=0)

    # Important to have consistent ordering!
    target_identifiers = mapping_file.index.map(str).values
    target_identifiers.sort()
    target_embeddings = list()

    with h5py.File(result_kwargs['reduced_embeddings_file'], 'r') as reduced_embeddings_file:
        for identifier in target_identifiers:
            target_embeddings.append(np.array(reduced_embeddings_file[identifier]))

    result_kwargs['n_jobs'] = result_kwargs.get('n_jobs', -1)
    result_kwargs['metric'] = result_kwargs.get('metric', 'euclidean')

    pairwise_distances = _pairwise_distances(
        target_embeddings,
        reference_embeddings,
        metric=result_kwargs['metric'],
        n_jobs=result_kwargs['n_jobs']
    )

    pairwise_distances_matrix_file_path = file_manager.create_file(result_kwargs.get('prefix'),
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

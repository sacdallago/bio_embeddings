import logging
from copy import deepcopy
from typing import Dict, Any, List

import h5py
import numpy as np
from Bio.Seq import Seq
from pandas import read_csv, DataFrame, concat as concatenate_dataframe
from sklearn.metrics import pairwise_distances as _pairwise_distances

from bio_embeddings.extract.basic import BasicAnnotationExtractor
from bio_embeddings.extract.light_attention.light_attention_annotation_extractor import \
    LightAttentionAnnotationExtractor
from bio_embeddings.extract.prott5cons import ProtT5consAnnotationExtractor
from bio_embeddings.extract.bindEmbed21.bindEmbed21DL_annotation_extractor import BindEmbed21DLAnnotationExtractor
from bio_embeddings.extract.bindEmbed21.bindEmbed21HBI_annotation_extractor import BindEmbed21HBIAnnotationExtractor
from bio_embeddings.extract.tmbed.tmbed_annotation_extractor import TmbedAnnotationExtractor
from bio_embeddings.extract.unsupervised_utilities import get_k_nearest_neighbours
from bio_embeddings.utilities import get_model_file, get_model_directories_from_zip
from bio_embeddings.utilities.exceptions import InvalidParameterError, UnrecognizedEmbeddingError, \
    InvalidAnnotationFileError
from bio_embeddings.utilities.filemanagers import get_file_manager
from bio_embeddings.utilities.helpers import check_required, read_fasta, convert_list_of_enum_to_string, \
    write_fasta_file, read_mapping_file

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
    reference_annotations_file = read_csv(result_kwargs["reference_annotations_file"])
    if not {"identifier", "label"} <= set(reference_annotations_file.columns):
        raise InvalidAnnotationFileError(
            f"The annotation file must contains columns 'identifier' and 'label', "
            f"but yours has {set(reference_annotations_file.columns)}"
        )

    # If reference annotations contain nans (either in label or identifier) throw an error!
    # https://github.com/sacdallago/bio_embeddings/issues/58
    # https://datatofish.com/check-nan-pandas-dataframe/
    if reference_annotations_file[['identifier', 'label']].isnull().values.any():
        raise InvalidAnnotationFileError(
            "Your annotation file contains NaN values in either identifier or label columns.\n"
            "Please remove these and run the pipeline again.")

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
    mapping_file = read_mapping_file(result_kwargs["mapping_file"])

    # Important to have consistent ordering!
    target_identifiers = mapping_file.index.values
    target_identifiers.sort()
    target_embeddings = list()

    with h5py.File(result_kwargs['reduced_embeddings_file'], 'r') as reduced_embeddings_file:
        for identifier in target_identifiers:
            target_embeddings.append(np.array(reduced_embeddings_file[identifier]))

    result_kwargs['n_jobs'] = result_kwargs.get('n_jobs', 1)
    result_kwargs['metric'] = result_kwargs.get('metric', 'euclidean')

    pairwise_distances = _pairwise_distances(
        target_embeddings,
        reference_embeddings,
        metric=result_kwargs['metric'],
        n_jobs=result_kwargs['n_jobs']
    )

    result_kwargs['keep_pairwise_distances_matrix_file'] = result_kwargs.get('keep_pairwise_distances_matrix_file',
                                                                             False)

    if result_kwargs['keep_pairwise_distances_matrix_file']:
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

    k_nn_indices, k_nn_distances = get_k_nearest_neighbours(pairwise_distances, result_kwargs['k_nearest_neighbours'])

    k_nn_identifiers = list(map(reference_identifiers.__getitem__, k_nn_indices))
    k_nn_annotations = list()

    for row in k_nn_identifiers:
        k_nn_annotations.append([";".join(
            reference_annotations_file[reference_annotations_file['identifier'] == identifier]['label'].values
        ) for identifier in row])

    # At this stage I have: nxk list of identifiers (strings), nxk indices (ints), nxk distances (floats),
    # nxk annotations
    # Now I need to expand the lists into a table and store the table into a CSV

    k_nn_identifiers_df = DataFrame(k_nn_identifiers,
                                    columns=[f"k_nn_{i + 1}_identifier" for i in range(len(k_nn_identifiers[0]))])
    k_nn_distances_df = DataFrame(k_nn_distances,
                                  columns=[f"k_nn_{i + 1}_distance" for i in range(len(k_nn_distances[0]))])
    k_nn_annotations_df = DataFrame(k_nn_annotations,
                                    columns=[f"k_nn_{i + 1}_annotations" for i in range(len(k_nn_annotations[0]))])

    transferred_annotations_dataframe = concatenate_dataframe(
        [k_nn_identifiers_df, k_nn_distances_df, k_nn_annotations_df], axis=1)
    transferred_annotations_dataframe.index = target_identifiers

    # At this stage we would like to aggregate all k_nn_XX_annotations into one column
    # -  A row in the k_nn_annotations matrix is string with X annotations (e.g. ["A;B", "A;C", "D"])
    # -  Each annotation in the string is separated by a ";"
    # Thus:
    # 1. Join all strings in a row separating them with ";" (aka ["A;B", "C"] --> "A;B;A;C;D")
    # 2. Split joined string into separate annotations using split(";") (aka "A;B;A;C;D" --> ["A","B","A","C","D"])
    # 3. Take a unique set of annotations by using set(*) (aka ["A","B","A","C","D"] --> set{"A","B","C","D"})
    # 4. Join the new unique set of annotations using ";" (aka set{"A","B","C","D"}) --> "A;B;C;D")
    transferred_annotations_dataframe['transferred_annotations'] = [";".join(set(";".join(k_nn_row).split(";"))) for
                                                                    k_nn_row in k_nn_annotations]

    # Merge with mapping file! Get also original ids!
    transferred_annotations_dataframe = mapping_file.join(transferred_annotations_dataframe)
    transferred_annotations_dataframe.to_csv(transferred_annotations_file_path, index=True)

    result_kwargs['transferred_annotations_file'] = transferred_annotations_file_path

    return result_kwargs


def seqvec_from_publication(**kwargs) -> Dict[str, Any]:
    return predict_annotations_using_basic_models("seqvec_from_publication", **kwargs)


def bert_from_publication(**kwargs) -> Dict[str, Any]:
    return predict_annotations_using_basic_models("bert_from_publication", **kwargs)


def t5_xl_u50_from_publication(**kwargs) -> Dict[str, Any]:
    return predict_annotations_using_basic_models("t5_xl_u50_from_publication", **kwargs)


def la_prott5(**kwargs) -> Dict[str, Any]:
    return light_attention('la_prott5', **kwargs)


def la_protbert(**kwargs) -> Dict[str, Any]:
    return light_attention('la_protbert', **kwargs)


def prott5cons(model: str, **kwargs) -> Dict[str, Any]:
    """
    Protocol extracts conservation from "embeddings_file".
    Embeddings can only be generated with ProtT5-XL-U50.

    :param model: "t5_xl_u50_conservation". Used to download files
    """

    check_required(kwargs, ['embeddings_file', 'mapping_file', 'remapped_sequences_file'])
    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager(**kwargs)

    # Download necessary files if needed
    for file in ProtT5consAnnotationExtractor.necessary_files:
        if not result_kwargs.get(file):
            result_kwargs[file] = get_model_file(model=model, file=file)

    annotation_extractor = ProtT5consAnnotationExtractor(**result_kwargs)

    # mapping file will be needed for protein-wide annotations
    mapping_file = read_mapping_file(result_kwargs["mapping_file"])

    # Try to create final files (if this fails, now is better than later
    conservation_predictions_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                  result_kwargs.get('stage_name'),
                                                                  'conservation_predictions_file',
                                                                  extension='.fasta')
    result_kwargs['conservation_predictions_file'] = conservation_predictions_file_path
    cons_sequences = list()
    with h5py.File(result_kwargs['embeddings_file'], 'r') as embedding_file:
        for protein_sequence in read_fasta(result_kwargs['remapped_sequences_file']):
            embedding = np.array(embedding_file[protein_sequence.id])

            annotations = annotation_extractor.get_conservation(embedding)
            cons_sequence = deepcopy(protein_sequence)
            cons_sequence.seq = Seq(convert_list_of_enum_to_string(annotations.conservation))
            cons_sequences.append(cons_sequence)

    # Write files
    write_fasta_file(cons_sequences, conservation_predictions_file_path)
    return result_kwargs


def bindembed21dl(**kwargs) -> Dict[str, Any]:
    """
    Protocol extracts binding residues from "embeddings_file".
    Results guaranteed only with ProtT5-XL-U50 embeddings.

    :return:
    """

    check_required(kwargs, ['embeddings_file', 'mapping_file', 'remapped_sequences_file'])
    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager(**kwargs)

    # Download necessary files if needed
    for file in BindEmbed21DLAnnotationExtractor.necessary_files:
        if not result_kwargs.get(file):
            result_kwargs[file] = get_model_file(model="bindembed21dl", file=file)

    annotation_extractor = BindEmbed21DLAnnotationExtractor(**result_kwargs)

    # Try to create final files (if this fails, now is better than later
    metal_binding_predictions_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                   result_kwargs.get('stage_name'),
                                                                   'metal_binding_predictions_file',
                                                                   extension='.fasta')
    result_kwargs['metal_binding_predictions_file'] = metal_binding_predictions_file_path
    nuc_binding_predictions_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                 result_kwargs.get('stage_name'),
                                                                 'nucleic_acid_binding_predictions_file',
                                                                 extension='.fasta')
    result_kwargs['binding_residue_predictions_file'] = nuc_binding_predictions_file_path
    small_binding_predictions_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                   result_kwargs.get('stage_name'),
                                                                   'small_molecule_binding_predictions_file',
                                                                   extension='.fasta')
    result_kwargs['binding_residue_predictions_file'] = small_binding_predictions_file_path

    metal_sequences = list()
    nuc_sequences = list()
    small_sequences = list()

    with h5py.File(result_kwargs['embeddings_file'], 'r') as embedding_file:
        for protein_sequence in read_fasta(result_kwargs['remapped_sequences_file']):
            embedding = np.array(embedding_file[protein_sequence.id])

            annotations = annotation_extractor.get_binding_residues(embedding)
            metal_sequence = deepcopy(protein_sequence)
            nuc_sequence = deepcopy(protein_sequence)
            small_sequence = deepcopy(protein_sequence)

            metal_sequence.seq = Seq(convert_list_of_enum_to_string(annotations.metal_ion))
            nuc_sequence.seq = Seq(convert_list_of_enum_to_string(annotations.nucleic_acids))
            small_sequence.seq = Seq(convert_list_of_enum_to_string(annotations.small_molecules))

            metal_sequences.append(metal_sequence)
            nuc_sequences.append(nuc_sequence)
            small_sequences.append(small_sequence)

    # Write files
    write_fasta_file(metal_sequences, metal_binding_predictions_file_path)
    write_fasta_file(nuc_sequences, nuc_binding_predictions_file_path)
    write_fasta_file(small_sequences, small_binding_predictions_file_path)

    return result_kwargs


def bindembed21hbi(**kwargs) -> Dict[str, Any]:
    """
    Protocol extracts binding residues from "alignment_results_file".

    :return:
    """

    check_required(kwargs, ['alignment_results_file', 'mapping_file', 'remapped_sequences_file'])
    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager(**kwargs)

    # Download necessary files if needed
    for directory in BindEmbed21HBIAnnotationExtractor.necessary_directories:
        if not result_kwargs.get(directory):
            result_kwargs[directory] = get_model_directories_from_zip(model="bindembed21hbi", directory=directory)

    annotation_extractor = BindEmbed21HBIAnnotationExtractor(**result_kwargs)

    # Try to create final files (if this fails, now is better than later
    metal_binding_predictions_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                   result_kwargs.get('stage_name'),
                                                                   'metal_binding_inference_file',
                                                                   extension='.fasta')
    result_kwargs['metal_binding_inference_file'] = metal_binding_predictions_file_path
    nuc_binding_predictions_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                 result_kwargs.get('stage_name'),
                                                                 'nucleic_acid_binding_inference_file',
                                                                 extension='.fasta')
    result_kwargs['binding_residue_inference_file'] = nuc_binding_predictions_file_path
    small_binding_predictions_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                   result_kwargs.get('stage_name'),
                                                                   'small_molecule_binding_inference_file',
                                                                   extension='.fasta')
    result_kwargs['binding_residue_inference_file'] = small_binding_predictions_file_path

    metal_sequences = list()
    nuc_sequences = list()
    small_sequences = list()

    alignment_results = read_csv(result_kwargs['alignment_results_file'], sep='\t',
                                 dtype={'query': 'str', 'target': 'str'})
    alignment_results = alignment_results[alignment_results['eval'] < 1E-3].copy()

    for protein_sequence in read_fasta(result_kwargs['remapped_sequences_file']):
        # get hits for this query
        hits = alignment_results[alignment_results['query'].str.match(str(protein_sequence.id))].copy()
        # get hits with minimal E-value
        hits_min_eval = hits[hits['eval'] == min(hits['eval'])]
        # get hit with maximal PIDE
        hit_max_pide = hits_min_eval[hits_min_eval['fident'] == max(hits_min_eval['fident'])]

        annotations = annotation_extractor.get_binding_residues(hit_max_pide.iloc[0].to_dict())
        metal_sequence = deepcopy(protein_sequence)
        nuc_sequence = deepcopy(protein_sequence)
        small_sequence = deepcopy(protein_sequence)

        metal_sequence.seq = Seq(convert_list_of_enum_to_string(annotations.metal_ion))
        nuc_sequence.seq = Seq(convert_list_of_enum_to_string(annotations.nucleic_acids))
        small_sequence.seq = Seq(convert_list_of_enum_to_string(annotations.small_molecules))

        metal_sequences.append(metal_sequence)
        nuc_sequences.append(nuc_sequence)
        small_sequences.append(small_sequence)

    # Write files
    write_fasta_file(metal_sequences, metal_binding_predictions_file_path)
    write_fasta_file(nuc_sequences, nuc_binding_predictions_file_path)
    write_fasta_file(small_sequences, small_binding_predictions_file_path)

    return result_kwargs


def bindembed21(**kwargs) -> Dict[str, Any]:
    """
    Protocol extracts binding residues from "alignment_result_file" if possible, and from "embeddings_file", otherwise.
    :param kwargs:
    :return:
    """

    check_required(kwargs, ['alignment_results_file', 'embeddings_file', 'mapping_file', 'remapped_sequences_file'])
    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager(**kwargs)

    # Download necessary files if needed
    # for HBI
    for directory in BindEmbed21HBIAnnotationExtractor.necessary_directories:
        if not result_kwargs.get(directory):
            result_kwargs[directory] = get_model_directories_from_zip(model="bindembed21hbi", directory=directory)
    # for DL
    for file in BindEmbed21DLAnnotationExtractor.necessary_files:
        if not result_kwargs.get(file):
            result_kwargs[file] = get_model_file(model="bindembed21dl", file=file)

    hbi_extractor = BindEmbed21HBIAnnotationExtractor(**result_kwargs)
    dl_extractor = BindEmbed21DLAnnotationExtractor(**result_kwargs)

    # Try to create final files (if this fails, now is better than later
    metal_binding_predictions_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                   result_kwargs.get('stage_name'),
                                                                   'metal_binding_predictions_file',
                                                                   extension='.fasta')
    result_kwargs['metal_binding_predictions_file'] = metal_binding_predictions_file_path
    nuc_binding_predictions_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                 result_kwargs.get('stage_name'),
                                                                 'nucleic_acid_binding_predictions_file',
                                                                 extension='.fasta')
    result_kwargs['binding_residue_predictions_file'] = nuc_binding_predictions_file_path
    small_binding_predictions_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                   result_kwargs.get('stage_name'),
                                                                   'small_molecule_binding_predictions_file',
                                                                   extension='.fasta')
    result_kwargs['binding_residue_predictions_file'] = small_binding_predictions_file_path

    metal_sequences = list()
    nuc_sequences = list()
    small_sequences = list()

    alignment_results = read_csv(result_kwargs['alignment_results_file'], sep='\t',
                                 dtype={'query': 'str', 'target': 'str'})
    alignment_results = alignment_results[alignment_results['eval'] < 1E-3].copy()

    with h5py.File(result_kwargs['embeddings_file'], 'r') as embedding_file:
        for protein_sequence in read_fasta(result_kwargs['remapped_sequences_file']):
            # get HBI hit for this query
            hits = alignment_results[alignment_results['query'].str.match(str(protein_sequence.id))].copy()
            hits_min_eval = hits[hits['eval'] == min(hits['eval'])]
            hit_max_pide = hits_min_eval[hits_min_eval['fident'] == max(hits_min_eval['fident'])]

            metal_sequence = deepcopy(protein_sequence)
            nuc_sequence = deepcopy(protein_sequence)
            small_sequence = deepcopy(protein_sequence)

            hbi_annotations = hbi_extractor.get_binding_residues(hit_max_pide.iloc[0].to_dict())
            metal_inference = convert_list_of_enum_to_string(hbi_annotations.metal_ion)
            nuc_inference = convert_list_of_enum_to_string(hbi_annotations.nucleic_acids)
            small_inference = convert_list_of_enum_to_string(hbi_annotations.small_molecules)

            # some part of the sequence was predicted using HBI --> save output and don't run DL method
            if 'M' in metal_inference or 'N' in nuc_inference or 'S' in small_inference:
                metal_sequence.seq = Seq(metal_inference)
                nuc_sequence.seq = Seq(nuc_inference)
                small_sequence.seq = Seq(small_inference)
            # no inference containing binding annotations was made --> run bindEmbed21DL
            else:
                embedding = np.array(embedding_file[protein_sequence.id])
                annotations = dl_extractor.get_binding_residues(embedding)
                metal_sequence = deepcopy(protein_sequence)
                nuc_sequence = deepcopy(protein_sequence)
                small_sequence = deepcopy(protein_sequence)

                metal_sequence.seq = Seq(convert_list_of_enum_to_string(annotations.metal_ion))
                nuc_sequence.seq = Seq(convert_list_of_enum_to_string(annotations.nucleic_acids))
                small_sequence.seq = Seq(convert_list_of_enum_to_string(annotations.small_molecules))

            metal_sequences.append(metal_sequence)
            nuc_sequences.append(nuc_sequence)
            small_sequences.append(small_sequence)

    # Write files
    write_fasta_file(metal_sequences, metal_binding_predictions_file_path)
    write_fasta_file(nuc_sequences, nuc_binding_predictions_file_path)
    write_fasta_file(small_sequences, small_binding_predictions_file_path)

    return result_kwargs


def tmbed(**kwargs) -> Dict[str, Any]:
    '''
    Protocol extracts membrane residues from "embeddings_file".
    Embeddings must have been generated with ProtT5-XL-U50.
    '''

    check_required(kwargs, ['embeddings_file', 'remapped_sequences_file'])

    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager(**kwargs)

    # Download necessary files if needed
    for file in TmbedAnnotationExtractor.necessary_files:
        if not result_kwargs.get(file):
            result_kwargs[file] = get_model_file(model='tmbed', file=file)

    tmbed_extractor = TmbedAnnotationExtractor(**result_kwargs)

    # Try to create final file (if this fails, now is better than later)
    membrane_residues_predictions_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                       result_kwargs.get('stage_name'),
                                                                       'membrane_residues_predictions_file',
                                                                       extension='.fasta')

    result_kwargs['membrane_residues_predictions_file'] = membrane_residues_predictions_file_path

    tmbed_sequences = []

    with h5py.File(result_kwargs['embeddings_file'], 'r') as embedding_file:
        for protein_sequence in read_fasta(result_kwargs['remapped_sequences_file']):
            embedding = np.array(embedding_file[protein_sequence.id])

            # Add batch dimension (until we support batch processing)
            embedding = embedding[None, ]

            # Sequence lengths (only a single sequence for now)
            lengths = [len(protein_sequence.seq)]

            annotations = tmbed_extractor.get_membrane_residues(embedding, lengths)

            # Gratuitous loop (only a single item for now)
            # Needs to be changed for batch mode to deepcopy different protein sequences
            for annotation in annotations:
                tmbed_sequence = deepcopy(protein_sequence)
                tmbed_sequence.seq = Seq(convert_list_of_enum_to_string(annotation.membrane_residues))

                tmbed_sequences.append(tmbed_sequence)

    # Write file
    write_fasta_file(tmbed_sequences, membrane_residues_predictions_file_path)

    return result_kwargs


def light_attention(model: str, **kwargs) -> Dict[str, Any]:
    """
    Protocol extracts subcellular locationfrom "embeddings_file".
    Embeddings can be generated with ProtBert.

    :param model: either "la_protbert" or "la_prott5". Used to download files
    """

    check_required(kwargs, ['embeddings_file', 'mapping_file', 'remapped_sequences_file'])
    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager(**kwargs)

    annotation_extractor = LightAttentionAnnotationExtractor(model, **result_kwargs)

    # mapping file will be needed for protein-wide annotations
    mapping_file = read_mapping_file(result_kwargs["mapping_file"])

    # Try to create final files (if this fails, now is better than later
    per_sequence_predictions_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                  result_kwargs.get('stage_name'),
                                                                  'per_sequence_predictions_file',
                                                                  extension='.csv')
    result_kwargs['per_sequence_predictions_file'] = per_sequence_predictions_file_path

    with h5py.File(result_kwargs['embeddings_file'], 'r') as embedding_file:
        for protein_sequence in read_fasta(result_kwargs['remapped_sequences_file']):
            embedding = np.array(embedding_file[protein_sequence.id])

            annotations = annotation_extractor.get_subcellular_location(embedding)

            # Per-sequence annotations, e.g. subcell loc & membrane boundness
            mapping_file.at[protein_sequence.id, 'subcellular_location'] = annotations.localization.value
            mapping_file.at[protein_sequence.id, 'membrane_or_soluble'] = annotations.membrane.value

    # Write files
    mapping_file.to_csv(per_sequence_predictions_file_path)

    return result_kwargs


def predict_annotations_using_basic_models(model: str, **kwargs) -> Dict[str, Any]:
    """
    Protocol extracts secondary structure (DSSP3 and DSSP8), disorder, subcellular location and membrane boundness
    from "embeddings_file". Embeddings can either be generated with SeqVec or ProtBert.
    SeqVec models are used in this publication: https://doi.org/10.1186/s12859-019-3220-8
    ProtTrans models are used in this publication: https://doi.org/10.1101/2020.07.12.199554

    :param model: either "bert_from_publication" or "seqvec_from_publication". Used to download files
    """

    check_required(kwargs, ['embeddings_file', 'mapping_file', 'remapped_sequences_file'])
    result_kwargs = deepcopy(kwargs)
    file_manager = get_file_manager(**kwargs)

    annotation_extractor = BasicAnnotationExtractor(model, **result_kwargs)

    # mapping file will be needed for protein-wide annotations
    mapping_file = read_mapping_file(result_kwargs["mapping_file"])

    # Try to create final files (if this fails, now is better than later)
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

    if 'get_activations' in kwargs and kwargs['get_activations']:
        DSSP3_raw_predictions_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                   result_kwargs.get('stage_name'),
                                                                   'DSSP3_raw_predictions_file',
                                                                   extension='.csv')
        result_kwargs['DSSP3_raw_predictions_file'] = DSSP3_raw_predictions_file_path
        DSSP8_raw_predictions_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                   result_kwargs.get('stage_name'),
                                                                   'DSSP8_raw_predictions_file',
                                                                   extension='.csv')
        result_kwargs['DSSP8_raw_predictions_file'] = DSSP8_raw_predictions_file_path
        disorder_raw_predictions_file_path = file_manager.create_file(result_kwargs.get('prefix'),
                                                                      result_kwargs.get('stage_name'),
                                                                      'disorder_raw_predictions_file',
                                                                      extension='.csv')
        result_kwargs['disorder_raw_predictions_file'] = disorder_raw_predictions_file_path

    # Create sequence containers
    DSSP3_sequences = list()
    DSSP8_sequences = list()
    disorder_sequences = list()

    DSSP3_raw = []
    DSSP8_raw = []
    disorder_raw = []

    with h5py.File(result_kwargs['embeddings_file'], 'r') as embedding_file:
        for protein_sequence in read_fasta(result_kwargs['remapped_sequences_file']):
            # Per-AA annotations: DSSP3, DSSP8 and disorder
            embedding = np.array(embedding_file[protein_sequence.id])

            annotations = annotation_extractor.get_annotations(embedding)

            DSSP3_sequence = deepcopy(protein_sequence)
            DSSP3_sequence.seq = Seq(convert_list_of_enum_to_string(annotations.DSSP3))
            DSSP3_sequences.append(DSSP3_sequence)
            DSSP3_raw_df = DataFrame(annotations.DSSP3_raw[:, :, 0].detach().cpu().numpy().transpose(),
                                     columns=['H', 'E', 'C'])
            DSSP3_raw_df.insert(0, 'residue', range(1, 1 + len(DSSP3_raw_df)))
            DSSP3_raw_df.insert(0, 'seqID', DSSP3_sequence.id)
            DSSP3_raw.append(DSSP3_raw_df)

            DSSP8_sequence = deepcopy(protein_sequence)
            DSSP8_sequence.seq = Seq(convert_list_of_enum_to_string(annotations.DSSP8))
            DSSP8_sequences.append(DSSP8_sequence)
            DSSP8_raw_df = DataFrame(annotations.DSSP8_raw[:, :, 0].detach().cpu().numpy().transpose(),
                                     columns=['G', 'H', 'I', 'B', 'E', 'S', 'T', 'C'])
            DSSP8_raw_df.insert(0, 'residue', range(1, 1 + len(DSSP8_raw_df)))
            DSSP8_raw_df.insert(0, 'seqID', DSSP8_sequence.id)
            DSSP8_raw.append(DSSP8_raw_df)

            disorder_sequence = deepcopy(protein_sequence)
            disorder_sequence.seq = Seq(convert_list_of_enum_to_string(annotations.disorder))
            disorder_sequences.append(disorder_sequence)
            disorder_raw_df = DataFrame(annotations.disorder_raw[:, :, 0].detach().cpu().numpy().transpose(),
                                        columns=['Order', 'Disorder'])
            disorder_raw_df.insert(0, 'residue', range(1, 1 + len(disorder_raw_df)))
            disorder_raw_df.insert(0, 'seqID', disorder_sequence.id)
            disorder_raw.append(disorder_raw_df)

            # Per-sequence annotations, e.g. subcell loc & membrane boundness
            mapping_file.at[protein_sequence.id, 'subcellular_location'] = annotations.localization.value
            mapping_file.at[protein_sequence.id, 'membrane_or_soluble'] = annotations.membrane.value

    # Write files
    mapping_file.to_csv(per_sequence_predictions_file_path)
    write_fasta_file(DSSP3_sequences, DSSP3_predictions_file_path)
    write_fasta_file(DSSP8_sequences, DSSP8_predictions_file_path)
    write_fasta_file(disorder_sequences, disorder_predictions_file_path)

    if 'get_activations' in kwargs and kwargs['get_activations']:
        # create files with activations for each multiclass prediction
        concatenate_dataframe(DSSP3_raw).set_index('seqID').rename_axis(None).to_csv(DSSP3_raw_predictions_file_path)
        concatenate_dataframe(DSSP8_raw).set_index('seqID').rename_axis(None).to_csv(DSSP8_raw_predictions_file_path)
        concatenate_dataframe(disorder_raw).set_index('seqID').rename_axis(None).to_csv(
            disorder_raw_predictions_file_path)

    return result_kwargs


# list of available extraction protocols
PROTOCOLS = {
    "bert_from_publication": bert_from_publication,
    "seqvec_from_publication": seqvec_from_publication,
    "t5_xl_u50_from_publication": t5_xl_u50_from_publication,
    "la_prott5": la_prott5,
    "la_protbert": la_protbert,
    "prott5cons": prott5cons,
    "bindembed21dl": bindembed21dl,
    "bindembed21hbi": bindembed21hbi,
    "bindembed21": bindembed21,
    "tmbed": tmbed,
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

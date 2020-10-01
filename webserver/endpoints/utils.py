from tempfile import NamedTemporaryFile
from pandas import read_csv
from flask import abort
from bio_embeddings.utilities import read_fasta
from webserver.utilities.configuration import configuration


def validate_FASTA_submission(request):
    files = request.files

    if 'sequences' not in files:
        return abort(400, "Missing files")

    # Test if sequences is valid FASTA, count number of AA & get identifiers
    AA_count = 0
    sequences_count = 0
    identifiers = set()

    try:
        file_io = files.get('sequences', {})
        temp_file = NamedTemporaryFile()
        file_io.save(temp_file.name)

        sequences = read_fasta(temp_file.name)
    except:
        return abort(400, "Could not read FASTA sequence")

    for sequence in sequences:
        if sequence.id in identifiers:
            return abort(400, "Your FASTA sequence contains duplicate identifiers (faulty identifier: {})".format(sequence.id))
        if not sequence.id:
            return abort(400, "Your FASTA sequence contains a sequence with no identifier. This is not allowed.")

        identifiers.add(sequence.id)
        AA_count += len(sequence)
        sequences_count += 1

        if AA_count > configuration['web']['max_amino_acids']:
            return abort(400, "Your FASTA file contains more than {}AA. The total allowed is {}AA. "
                              "Please, exclude some sequences from your file "
                              "or consider running the bio_embeddings pipeline locally.".format(AA_count, configuration['web']['max_amino_acids']))

    # Compile statistics
    statistics = dict(
        numberOfSequences=sequences_count,
        numberOfAA=AA_count,
    )

    if statistics['numberOfSequences'] < 1:
        return abort(400, "No sequences submitted. Try another FASTA file.")

    result = dict(
        sequences=sequences,
        statistics=statistics
    )

    return result


def validate_file_submission(request):
    files = request.files

    if 'sequences' not in files or 'annotations' not in files:
        return abort(400, "Missing files")

    # Test if sequences is valid FASTA, count number of AA & get identifiers
    AA_count = 0
    sequences_count = 0
    identifiers = set()

    try:
        file_io = files.get('sequences', {})
        temp_file = NamedTemporaryFile()
        file_io.save(temp_file.name)

        sequences = read_fasta(temp_file.name)
    except:
        return abort(400, "Could not read FASTA sequence")

    for sequence in sequences:
        if sequence.id in identifiers:
            return abort(400, "Your FASTA sequence contains duplicate identifiers (faulty idenfifier: {})".format(sequence.id))
        if not sequence.id:
            return abort(400, "Your FASTA sequence contains a sequence with no identifier. This is not allowed.")

        identifiers.add(sequence.id)
        AA_count += len(sequence)
        sequences_count += 1

        if AA_count > configuration['web']['max_amino_acids']:
            return abort(400, "Your FASTA file contains more than {}AA. The total allowed is {}AA. "
                              "Please, exclude some sequences from your file "
                              "or consider running the bio_embeddings pipeline locally.".format(AA_count, configuration['web']['max_amino_acids']))

    # Test if annotations file is valid CSV and contains necessary columns
    try:
        file_io = files.get('annotations', {})
        file_io.save(temp_file.name)

        annotations = read_csv(temp_file.name, index_col='identifier')
    except:
        return abort(400, "Could not read annotations CSV. "
                                     "Make sure it's a CSV file with header (identifier,label).")

    if not annotations.index.is_unique:
        return abort(400, "Your CSV has multiple annotations for the same identifier. ")

    # Generate statistics about overlapping annotations
    identifiers_from_annotations = set(annotations.index.values)
    indexes_intersection = identifiers & identifiers_from_annotations
    indexes_intersection = list(indexes_intersection)

    if len(indexes_intersection) is 0:
        return abort(400, "There is no overlap between FASTA identifiers and annotation file identifiers.")

    # Compile statistics
    statistics = dict(
        numberOfSequences=sequences_count,
        numberOfAA=AA_count,
        numberOfAnnotatedSequences=len(indexes_intersection),
        annotatedIdentifiers=indexes_intersection
    )

    result = dict(
        sequences=sequences,
        annotations=annotations,
        statistics=statistics
    )

    return result

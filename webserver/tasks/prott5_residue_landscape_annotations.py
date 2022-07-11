import io
import logging

from webserver.tasks import task_keeper
from webserver.utilities.configuration import configuration

if 'prott5_residue_landscape_annotations' in configuration['celery']['celery_worker_type']:
    from pathlib import Path

    import h5py
    import numpy as np

    from vespa.predict.config import MODEL_PATH_DICT
    from vespa.predict.conspred import ProtT5Cons, get_dataloader
    from vespa.predict.vespa import VespaPred
    from vespa.predict.utils import MutationGenerator
    from vespa.predict.config import MUTANT_ORDER


@task_keeper.task()
def get_residue_landscape_output_sync(sequence: str, embedding_as_list: list) -> dict:
    # method takes the embeddings as input and returns the conspred and residue_landscape

    ################################################conspred################################################
    checkpoint_path = Path(MODEL_PATH_DICT["CONSCNN"])
    write_probs = True
    write_classes = True

    embedding_buffer = io.BytesIO()
    with h5py.File(embedding_buffer, "w") as embeddings_file:
        embeddings_file.create_dataset("sequence", data=np.array(embedding_as_list))

    embedding_buffer.seek(0)

    # this could be replaced if another method get_dataLoader from dict is implemented
    embeddings = h5py.File(embedding_buffer, 'r')
    data_loader = get_dataloader(embeddings, batch_size=128)
    # Vespa has no methods that allow sharing python objects between scripts

    conspred = ProtT5Cons(checkpoint_path)

    # hard coded to only out put propabilities for now

    predictions = conspred.conservation_prediction(
        data_loader, prob_return=write_probs, class_return=write_classes
    )

    cons_probs = predictions.probability_out
    classes_out = predictions.class_out

    ################################################conspred################################################
    ################################################Vespa################################################

    seq_dict = {'sequence': sequence}

    mutation_file = None
    one_based_mutations = None

    mutation_gen = MutationGenerator(
        sequence_dict=seq_dict,
        file_path=mutation_file,
        one_based_file=one_based_mutations,
    )

    predictor = VespaPred(vespa=False, vespal=True)

    input = dict()
    input["conservations"] = cons_probs

    vespal_output = predictor.generate_predictions(mutation_gen=mutation_gen, **input)

    embeddings.close()

    cons_probs_arr = np.array(cons_probs['sequence']) * 100
    cons_probs_arr = cons_probs_arr.astype(np.int8)

    # build residue_landscape return dict from dictionary
    # initialize the matrix that will be build column wise
    # counts how many amino acids are inserted into the column already
    counter_aa = 0
    # keeps track of the position on the x axis
    pos_seq = 0
    # a new column that will be added to the matirx
    new_column = []
    # the actual matrix
    matrix = []
    # a marker that tracks if the values for self substitution has been added eg. if  x=M y=M has been set to -1
    marker = False
    # fill  matrix column wise
    # the mutant is in the for <AA on seq><pos in seq><substitution AA> eg. M4A
    for mutant, vespa_score in vespal_output['sequence']:
        residue = mutant[0]
        position = int(mutant[1:-1])
        substitute_aa = mutant[-1]

        # assert statements to make sure that the postion into which the value is inserted matched the read mutation
        # check if the x axis postion matches the postion from the mutant annotation
        assert pos_seq == position, f'Mutation position and writing position didnt match {position} {pos_seq} {counter_aa}'
        # check if the read residue actually is at the postion at which the value will be inserted
        assert residue == sequence[pos_seq], 'Residues didn\'t match'

        # assert if the aa that is substituted is inserted at the right postion but only if its not the same as the one on the x axis
        if MUTANT_ORDER[counter_aa] != residue:
            assert substitute_aa == MUTANT_ORDER[counter_aa], f'Wrong AA should be inserted {position} {pos_seq} {counter_aa} '
        else:
            # if the residue on the x axis is the same as the aa on the y axis insert -1
            marker = True
            new_column.append(-1)
            counter_aa += 1

        # transform value to int8 for better storage
        value = int(vespa_score['VESPAl'] * 100)
        # add value to column
        new_column.append(value)
        counter_aa += 1

        # if the counter for the AA is a 20 an entire column has been filled and the counter moves to the next postion on the x-axis
        if counter_aa == 20:
            matrix.append(new_column)
            new_column = []
            counter_aa = 0
            pos_seq += 1
            marker = False

        # corner case for when the residue on the x axis is the last one on the mutant order
        if not marker and counter_aa == 19:
            new_column.append(-1)
            matrix.append(new_column)
            new_column = []
            counter_aa = 0
            pos_seq += 1
            marker = False

    # create an ndarray to transpose the matrix since it has been build col wise

    matrix_transposed = np.array(matrix).T
    matrix_transposed = matrix_transposed.astype(np.int8).tolist()

    x_axis_seq = [res for res in sequence]
    y_axis_Mutant = [aa for aa in MUTANT_ORDER]

    vespa_return_dict = {
        'x_axis': x_axis_seq,
        'y_axis': y_axis_Mutant,
        'values': matrix_transposed
    }
    conservation_return_dict = {
        'x_axis': x_axis_seq,
        'y_axis': y_axis_Mutant,
        'values': cons_probs_arr.tolist()
    }

    return {
        'predictedConservation': classes_out['sequence'].tolist(),
        'predictedVariation': vespa_return_dict,
        'meta': {
            'predictedConservation': 'https://link.springer.com/article/10.1007/s00439-021-02411-y',
            'predictedVariation': 'https://link.springer.com/article/10.1007/s00439-021-02411-y'
            }
        }

import logging
import h5py
import io
from typing import Tuple
import numpy as np
from webserver.tasks import task_keeper
from webserver.utilities.configuration import configuration
from pathlib import Path

if 'vespa' in configuration['celery']['celery_worker_type']:
    from vespa.predict.config import MODEL_PATH_DICT
    from vespa.predict.conspred import ProtT5Cons, get_dataloader
    import torch.utils.data
    from vespa.predict import utils
    from vespa.predict.vespa import *
    from vespa.predict.config import MUTANT_ORDER

    from celery.contrib import rdb


@task_keeper.task()
def get_vespa_output_sync(sequence: str, embedding_as_list: list):
    # method takes the embeddings as input and returns the conspred and vespa

    ################################################conspred################################################
    checkpoint_path = Path(MODEL_PATH_DICT["CONSCNN"])
    write_probs = True
    write_classes = False

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

    mutation_gen = utils.MutationGenerator(
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


    ## build vespa return
    counter_aa = 0
    pos_seq = 0
    new_column = []
    matrix = []
    marker = False
    # fill  matrix column wise
    for mutant, vespa_score in vespal_output['sequence']:
        residue = mutant[0]
        position = int(mutant[1:-1])
        substitute_aa = mutant[-1]

        assert pos_seq == position, f'Mutation position and writing position didnt match {position} {pos_seq} {counter_aa}'
        assert residue == sequence[pos_seq], 'Residues didn\'t match'
        if MUTANT_ORDER[counter_aa] != residue:
            assert substitute_aa == MUTANT_ORDER[counter_aa], f'Wrong AA should be inserted {position} {pos_seq} {counter_aa} '
        else:
            marker = True
            new_column.append(-1)
            counter_aa += 1

        value = int(vespa_score['VESPAl'] * 100)

        new_column.append(value)
        counter_aa += 1

        if counter_aa == 20:
            matrix.append(new_column)
            new_column = []
            counter_aa = 0
            pos_seq += 1
            marker = False

        if not marker and counter_aa == 19:
            new_column.append(-1)
            matrix.append(new_column)
            new_column = []
            counter_aa = 0
            pos_seq += 1
            marker = False

    matrix_transposed = np.array(matrix).T
    matrix_transposed = matrix_transposed.astype(np.int8).tolist()

    vespa_return_dict = {'x_axis': [res for res in sequence],
                         'y_axis': [aa for aa in MUTANT_ORDER],
                         'values': matrix_transposed}

    #rdb.set_trace()

    return cons_probs_arr.tolist(), vespa_return_dict

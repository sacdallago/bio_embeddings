from colabfold.batch import get_queries, set_model_type, run
from colabfold.download import download_alphafold_params
from hashlib import md5
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict
import os.path

import logging
from webserver.tasks import task_keeper
from webserver.utilities.configuration import configuration


MAX_SEQUENCE_LENGTH = 500

if "colabfold" in configuration['celery']['celery_worker_type']:
    logger = logging.getLogger()

    predictions_dir = TemporaryDirectory(prefix='colabfold-predictions')


@task_keeper.task()
def get_structure_colabfold(query_sequence: str) -> Dict[str, str]:
    """
    Args:
        query_sequence: Sequence to predict structure for

    Returns:
        TODO

     Predicts the structure of a protein sequence by calling the colabfold batch processing method.
     Basically a copy of the ColabFold notebook at
     https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb#scrollTo=kOblAo-xetgx
     The parameters here are almost exactly set to the values of the ColabFold Default parameters.
    """

    # Sequence input and setup
    if len(query_sequence) > MAX_SEQUENCE_LENGTH:
        return {
            'sequence': query_sequence,
            'result': f'Query sequence too long. Must not exceed {MAX_SEQUENCE_LENGTH}'
        }

    sequence_hash = md5(query_sequence.encode())
    job_id = sequence_hash.hexdigest()

    queries_path = f"query{sequence_hash.hexdigest()}.csv"
    with open(queries_path, 'w') as queries_file:
        queries_file.write(f"id,sequence\n{job_id},{query_sequence}")

    result_dir = os.path.join(predictions_dir.name, job_id)
    try:
        os.mkdir(result_dir)
    except FileExistsError:
        return {
            'sequence': query_sequence,
            'result': 'Query already in progress'
        }

    use_amber = False
    custom_template_path = None
    use_templates = False

    # MSA Options
    msa_mode = "MMseqs2 (UniRef+Environmental)"
    pair_mode = "unpaired+paired"

    # Advanced Settings
    model_type = "auto"
    num_recycles = 3

    # Preparation for run
    def prediction_callback(unrelaxed_protein, length, prediction_result, input_features, type):
        logger.info("Finished prediction")

    queries, is_complex = get_queries(queries_path)
    model_type = set_model_type(is_complex, model_type)
    download_alphafold_params(model_type, Path("."))

    # Run structure prediction
    run(
        queries=queries,
        result_dir=result_dir,
        use_templates=use_templates,
        custom_template_path=custom_template_path,
        use_amber=use_amber,
        msa_mode=msa_mode,
        model_type=model_type,
        num_models=1,
        num_recycles=num_recycles,
        model_order=[3],
        is_complex=is_complex,
        data_dir=Path("."),
        keep_existing_results=False,
        recompile_padding=1.0,
        rank_by="auto",
        pair_mode=pair_mode,
        stop_at_score=float(85),
        stop_at_score_below=float(40),
        prediction_callback=prediction_callback,
    )

    return {
        'result': "success",
        'sequence': query_sequence,
        'result_dir': result_dir,
        'msa_file': os.path.join(result_dir, f"{job_id}.a3m"),
        'pdb_file': os.path.join(result_dir, f"{job_id}_unrelaxed_rank_1_model_3.pdb"),
        'json_file': os.path.join(result_dir, f"{job_id}_unrelaxed_rank_1_model_3_scores.json"),
    }

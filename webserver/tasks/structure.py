from typing import Dict

from webserver.tasks import task_keeper
from webserver.utilities.configuration import configuration

if "colabfold" in configuration['celery']['celery_worker_type']:
    import os.path
    from colabfold.batch import get_queries, set_model_type, run
    from datetime import datetime
    from hashlib import md5
    from tempfile import TemporaryDirectory

    from webserver.database import get_structure_cache, get_structure_jobs, JOB_PENDING, JOB_DONE, JOB_FAILED


MAX_SEQUENCE_LENGTH = 500


def _get_structure_colabfold(query_sequence: str) -> Dict[str, object]:
    # Sequence input and setup
    if len(query_sequence) > MAX_SEQUENCE_LENGTH:
        return {
            'result': f"Query sequence too long. Must not exceed {MAX_SEQUENCE_LENGTH}",
            'sequence': query_sequence,
        }

    sequence_hash = md5(query_sequence.encode())
    job_id = sequence_hash.hexdigest()

    with TemporaryDirectory(prefix='colabfold', suffix=job_id) as result_dir:
        queries_path = os.path.join(result_dir, "query.csv")
        with open(queries_path, 'w') as queries_file:
            queries_file.write(f"id,sequence\n{job_id},{query_sequence}")

        # Advanced Settings
        queries, is_complex = get_queries(queries_path)
        model_type = set_model_type(is_complex, "auto")

        # Run structure prediction
        run(
            queries=queries,
            result_dir=result_dir,
            use_templates=False,
            custom_template_path=None,
            use_amber=False,
            msa_mode="MMseqs2 (UniRef+Environmental)",
            model_type=model_type,
            num_models=1,
            num_recycles=3,
            model_order=[3],
            is_complex=is_complex,
            data_dir=os.path.join(configuration['colabfold']['data_dir']),
            keep_existing_results=False,
            recompile_padding=1.0,
            rank_by="auto",
            pair_mode="unpaired+paired",
            stop_at_score=float(85),
            stop_at_score_below=float(40),
        )

        with open(os.path.join(result_dir, f"{job_id}.a3m"), 'r') as msa_file:
            msa = msa_file.read()
        with open(os.path.join(result_dir, f"{job_id}_unrelaxed_rank_1_model_3.pdb"), 'r') as pdb_file:
            pdb = pdb_file.read()
        with open(os.path.join(result_dir, f"{job_id}_unrelaxed_rank_1_model_3_scores.json"), 'r') as json_file:
            json = json_file.read()

    return {
        'msa': msa,
        'pdb': pdb,
        'json': json,
        'meta': {
            'msa': "ColabFold, https://www.nature.com/articles/s41592-022-01488-1",
            'pdb': "ColabFold, https://www.nature.com/articles/s41592-022-01488-1",
            'json': "ColabFold:, https://www.nature.com/articles/s41592-022-01488-1",
        }
    }


@task_keeper.task()
def get_structure_colabfold(query_sequence: str):
    """
    Args:
        query_sequence: Sequence to predict structure for

     Predicts the structure of a protein sequence by calling the colabfold batch processing method.
     Based on the colabfold notebook at:
     https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb#scrollTo=kOblAo-xetgx
     The parameters here are almost exactly set to the values of the ColabFold Default parameters.
    """

    # In case that there is already a job for this structure, we don't need to do anything
    in_progress = get_structure_jobs.find_one({'predictor_name': 'colabfold', 'sequence': query_sequence})
    if in_progress:
        return

    # Otherwise, we start a job and insert a corresponding entry in the database
    get_structure_jobs.insert_one({
        'timestamp': datetime.utcnow(),
        'predictor_name': 'colabfold',
        'sequence': query_sequence,
        'status': JOB_PENDING,
    })

    try:
        structure = _get_structure_colabfold(query_sequence)

        # Insert the structure into the structure cache and then update job database
        # The job should only be marked as done if there is a structure in the database
        get_structure_cache.insert_one({
            'uploadDate': datetime.utcnow(),
            'predictor_name': 'colabfold',
            'sequence': query_sequence,
            'structure': structure,
        })
        get_structure_jobs.update_one(
            {
                'sequence': query_sequence,
                'status': JOB_PENDING,
            },
            {
                '$set':
                    {
                        'timestamp': datetime.utcnow(),
                        'status': JOB_DONE,
                    }
            }
        )

    # If the job fails, no matter the reason, we want to make sure that the job database stays consistent
    except Exception as e:
        get_structure_jobs.update_one(
            {
                'sequence': query_sequence,
                'status': JOB_PENDING,
            },
            {
                '$set':
                    {
                        'timestamp': datetime.utcnow(),
                        'status': JOB_FAILED,
                    }
            }
        )
        raise e

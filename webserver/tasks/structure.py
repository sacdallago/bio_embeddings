from colabfold.batch import get_queries, set_model_type, run
from colabfold.download import download_alphafold_params
from hashlib import md5
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict
import os.path

import logging
from webserver.tasks import task_keeper


MAX_SEQUENCE_LENGTH = 500

logger = logging.getLogger()


@task_keeper.task()
def get_structure_colabfold(query_sequence: str) -> Dict[str, object]:
    """
    Args:
        query_sequence: Sequence to predict structure for

    Returns:
        TODO

     Predicts the structure of a protein sequence by calling the colabfold batch processing method.
     Based on the colabfold notebook at:
     https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb#scrollTo=kOblAo-xetgx
     The parameters here are almost exactly set to the values of the ColabFold Default parameters.
    """

    # Sequence input and setup
    if len(query_sequence) > MAX_SEQUENCE_LENGTH:
        return {
            'result': f"Query sequence too long. Must not exceed {MAX_SEQUENCE_LENGTH}",
            'sequence': query_sequence,
        }

    # TODO: Check if job is in progress
    sequence_hash = md5(query_sequence.encode())
    job_id = sequence_hash.hexdigest()

    queries_path = f"query{sequence_hash.hexdigest()}.csv"
    with open(queries_path, 'w') as queries_file:
        queries_file.write(f"id,sequence\n{job_id},{query_sequence}")

    use_amber = False
    custom_template_path = None
    use_templates = False

    # MSA Options
    msa_mode = "MMseqs2 (UniRef+Environmental)"
    pair_mode = "unpaired+paired"

    # Advanced Settings
    model_type = "auto"
    num_recycles = 3

    queries, is_complex = get_queries(queries_path)
    model_type = set_model_type(is_complex, model_type)
    download_alphafold_params(model_type, Path("."))

    with TemporaryDirectory(prefix='colabfold', suffix=job_id) as result_dir:
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

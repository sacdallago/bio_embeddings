from colabfold.batch import get_queries, set_model_type, run
from colabfold.colabfold import plot_protein
from colabfold.download import download_alphafold_params
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

import logging
from webserver.tasks import task_keeper

logger = logging.getLogger()


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

    job_id = 0
    # Sequence input and setup
    query_sequence = "".join(query_sequence.split())
    queries_path = "test.csv"
    with open(queries_path, 'w') as queries_file:
        queries_file.write(f"id,sequence\njob{job_id},{query_sequence}")

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
        fig = plot_protein(unrelaxed_protein, Ls=length, dpi=100)
        plt.show()
        plt.close()

    result_dir = "."
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
        model_order=[1],
        is_complex=is_complex,
        data_dir=Path("."),
        keep_existing_results=False,
        recompile_padding=1.0,
        rank_by="auto",
        pair_mode=pair_mode,
        stop_at_score=float(100),
        prediction_callback=prediction_callback,
    )

    return {
        'sequence': query_sequence,
        'result_path': '.'
    }


import os
from typing import Callable, Union

import numpy
import pytest
from tqdm import tqdm

from bio_embeddings.embed import (
    EmbedderInterface,
    ProtTransT5XLU50Embedder,
)

from bio_embeddings.extract.prott5cons import ProtT5consAnnotationExtractor
from bio_embeddings.utilities import read_fasta, get_model_file

# >5dzo-A
FASTA = """
MVKVGGEAGPSVTLPCHYSGAVTSMCWNRGSCSLFTCQNGIVWTNGTHVTYRKDTRYKLLGDLSRRDVSLTIENTAVSDSGVYCCRVEHRGWFNDMKITVSLEIVPP
"""
# groundtruth scores from ConSurf-DB
Y = """
6,5,1,8,3,8,1,4,9,4,4,7,6,9,4,9,1,8,5,1,3,1,6,3,6,9,9,7,6,5,1,6,6,1,3,3,9,3,4,3,6,6,4,7,6,5,1,1,7,3,3,4,1,7,2,8,9,4,8,1,7,2,5,1,1,8,7,7,9,9,9,9,1,5,5,3,4,2,9,6,9,1,9,8,9,9,7,6,6,4,9,1,8,9,9,3,7,1,4,3,1,9,3,7,1,9,6
""".split(',')

@pytest.mark.skipif(
    not os.environ.get("RUN_VERY_SLOW_TESTS"),
    reason="This checks the entire hard test set",
)
@pytest.mark.parametrize(
    "get_embedder,get_extractor,expected_accuracy",
    [
        (
            lambda: ProtTransT5XLU50Embedder(half_precision_model=True),
            lambda: ProtT5consAnnotationExtractor("prott5cons"),
            0.262,
        ),
    ],
)
def test_conservation_annotation_extractor(
    pytestconfig,
    get_embedder: Callable[[], EmbedderInterface],
    get_extractor: Callable[
        [], Union[ProtT5consAnnotationExtractor]
    ],
    expected_accuracy: float,
):
    """Check that BasicAnnotationExtractor passes (without checking correctness)"""
    extractor = get_extractor()
    embedder = get_embedder()

    embedding = embedder.embed(FASTA)
    prediction = extractor.get_conservation(embedding)
    results = [1 if groundtruth == prediction.conservation[idx] else 0 for idx, groundtruth in enumerate(Y) ]

    actual_accuracy = sum(results)/len(results)
    assert actual_accuracy == pytest.approx(expected_accuracy)

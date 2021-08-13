
import os
from typing import Callable, Union

import pytest

from bio_embeddings.embed import (
    EmbedderInterface,
    ProtTransT5XLU50Embedder,
)

from bio_embeddings.extract.prott5cons import ProtT5consAnnotationExtractor

# >5dzo-A
FASTA = """
MVKVGGEAGPSVTLPCHYSGAVTSMCWNRGSCSLFTCQNGIVWTNGTHVTYRKDTRYKLLGDLSRRDVSLTIENTAVSDSGVYCCRVEHRGWFNDMKITVSLEIVPP
""".strip()
# groundtruth scores from ConSurf-DB
Y = list("65183814944769491851316369976516613393436647651173341728948172511877999915534296919899766491899371431937196")

@pytest.mark.skipif(
    os.environ.get("SKIP_SLOW_TESTS"), 
    reason="This test is very slow",
)
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
    results = [actual.value == predicted for actual, predicted in zip(prediction.conservation, Y)]
    
    actual_accuracy = sum(results)/len(results)
    assert actual_accuracy == pytest.approx(expected_accuracy)

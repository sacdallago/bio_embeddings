import os
from typing import Callable, Union

import pytest
import torch

from bio_embeddings.embed import EmbedderInterface, ProtTransT5XLU50Embedder
from bio_embeddings.extract.prott5cons import ProtT5consAnnotationExtractor

# >5dzo-A
FASTA = """
MVKVGGEAGPSVTLPCHYSGAVTSMCWNRGSCSLFTCQNGIVWTNGTHVTYRKDTRYKLLGDLSRRDVSLTIENTAVSDSGVYCCRVEHRGWFNDMKITVSLEIVPP
""".strip()
# ground truth scores from ConSurf-DB
ground_truth = list(
    "65183814944769491851316369976516613393436647651173341728948172511877999915534296919899766491899371431937196"
)


@pytest.mark.skipif(os.environ.get("SKIP_SLOW_TESTS"), reason="Uses T5")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="T5 fp16 needs a GPU")
@pytest.mark.parametrize(
    "get_embedder,get_extractor,expected_accuracy",
    [
        (
            lambda: ProtTransT5XLU50Embedder(half_precision_model=True),
            lambda: ProtT5consAnnotationExtractor("prott5cons"),
            0.262,
        )
    ],
)
def test_conservation_annotation_extractor(
    get_embedder: Callable[[], EmbedderInterface],
    get_extractor: Callable[[], Union[ProtT5consAnnotationExtractor]],
    expected_accuracy: float,
):
    """Check that BasicAnnotationExtractor passes (without checking correctness)"""
    extractor = get_extractor()
    embedder = get_embedder()

    embedding = embedder.embed(FASTA)
    prediction = extractor.get_conservation(embedding)
    results = [
        actual.value == predicted
        for actual, predicted in zip(prediction.conservation, ground_truth)
    ]

    actual_accuracy = sum(results) / len(results)
    assert actual_accuracy == pytest.approx(expected_accuracy)

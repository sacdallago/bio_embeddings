import os
from typing import Callable, Union

import pytest

from bio_embeddings.embed import (
    EmbedderInterface,
    ProtTransT5XLU50Embedder,
)

from bio_embeddings.extract.bindEmbed21DL import BindEmbed21DLAnnotationExtractor

# >P58568
FASTA = 'MADKADQSSYLIKFISTAPVAATIWLTITAGILIEFNRFFPDLLFHPLP'
# groundtruth values for small molecule binding
Y = list("0000000000000000001011111101111110111110011101010")


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
                lambda: BindEmbed21DLAnnotationExtractor("bindembed21"),
                0.694,
        ),
    ],
)
def test_binding_residue_annotation_extractor(
        get_embedder: Callable[[], EmbedderInterface],
        get_extractor: Callable[
            [], Union[BindEmbed21DLAnnotationExtractor]
        ],
        expected_accuracy: float,
):
    """Check that BasicAnnotationExtractor passes (without checking correctness)"""
    extractor = get_extractor()
    embedder = get_embedder()

    embedding = embedder.embed(FASTA)
    prediction = extractor.get_binding_residues(embedding)

    results = [actual.value == predicted for actual, predicted in zip(prediction.small_molecules, Y)]

    actual_accuracy = sum(results) / len(results)
    assert actual_accuracy == pytest.approx(expected_accuracy)

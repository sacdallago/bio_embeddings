import os
from typing import Callable, Union

import pytest
import torch

from bio_embeddings.embed import EmbedderInterface, ProtTransT5XLU50Embedder
from bio_embeddings.extract.bindEmbed21 import BindEmbed21DLAnnotationExtractor

# >P58568
FASTA = "MADKADQSSYLIKFISTAPVAATIWLTITAGILIEFNRFFPDLLFHPLP"
# groundtruth values for small molecule binding
Y = list("------------------S-SSSSSS-SSSSSS-SSSSS--SSS-S-S-")


@pytest.mark.skipif(os.environ.get("SKIP_SLOW_TESTS"), reason="Uses T5")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="T5 fp16 needs a GPU")
@pytest.mark.parametrize(
    "get_embedder,get_extractor,expected_accuracy",
    [
        (
            lambda: ProtTransT5XLU50Embedder(half_precision_model=True),
            lambda: BindEmbed21DLAnnotationExtractor(),
            0.714,
        )
    ],
)
def test_binding_residue_annotation_extractor(
    get_embedder: Callable[[], EmbedderInterface],
    get_extractor: Callable[[], Union[BindEmbed21DLAnnotationExtractor]],
    expected_accuracy: float,
):
    """Check that BasicAnnotationExtractor passes (without checking correctness)"""
    extractor = get_extractor()
    embedder = get_embedder()

    embedding = embedder.embed(FASTA)
    prediction = extractor.get_binding_residues(embedding)

    results = [
        actual.value == predicted
        for actual, predicted in zip(prediction.small_molecules, Y)
    ]

    actual_accuracy = round(sum(results) / len(results), 3)
    assert actual_accuracy == pytest.approx(expected_accuracy)

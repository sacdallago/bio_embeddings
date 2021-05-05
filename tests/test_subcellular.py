import os
from typing import Callable, Union

import numpy
import pytest
from tqdm import tqdm

from bio_embeddings.embed import (
    EmbedderInterface,
    ProtTransT5XLU50Embedder,
    SeqVecEmbedder,
    ProtTransBertBFDEmbedder,
)
from bio_embeddings.extract import BasicAnnotationExtractor
from bio_embeddings.extract.light_attention import LightAttentionAnnotationExtractor
from bio_embeddings.utilities import read_fasta, get_model_file


def normalize_location(localization: str) -> str:
    """Quick and dirty normalization because the test set and bio_embeddings internally use slightly different terms"""
    return localization.replace(".", "").replace("-", "").replace(" ", "").lower()


@pytest.mark.skipif(
    not os.environ.get("RUN_VERY_SLOW_TESTS"),
    reason="This checks the entire hard test set",
)
@pytest.mark.parametrize(
    "get_embedder,get_extractor,expected_accuracy",
    [
        (
            lambda: SeqVecEmbedder(),
            lambda: BasicAnnotationExtractor("seqvec_from_publication"),
            0.5142857142857142,
        ),
        (
            lambda: ProtTransBertBFDEmbedder(),
            lambda: BasicAnnotationExtractor("bert_from_publication"),
            0.5387755102040817,
        ),
        (
            lambda: ProtTransT5XLU50Embedder(half_precision_model=True),
            lambda: BasicAnnotationExtractor("t5_xl_u50_from_publication"),
            0.6285714285714286,
        ),
        (
            lambda: ProtTransT5XLU50Embedder(half_precision_model=True),
            lambda: LightAttentionAnnotationExtractor(
                subcellular_location_checkpoint_file=get_model_file(
                    "la_prott5", "subcellular_location_checkpoint_file"
                ),
                membrane_checkpoint_file=get_model_file(
                    "la_prott5", "membrane_checkpoint_file"
                ),
            ),
            0.6551020408163265,
        ),
    ],
)
def test_basic_annotation_extractor(
    pytestconfig,
    get_embedder: Callable[[], EmbedderInterface],
    get_extractor: Callable[
        [], Union[BasicAnnotationExtractor, LightAttentionAnnotationExtractor]
    ],
    expected_accuracy: float,
):
    """Check that BasicAnnotationExtractor passes (without checking correctness)"""
    extractor = get_extractor()
    embedder = get_embedder()

    results = []
    new_hard_set = pytestconfig.rootpath.joinpath(
        "test-data/subcellular_location_new_hard_set.fasta"
    )
    for record in tqdm(read_fasta(str(new_hard_set))):
        embedding = embedder.embed(str(record.seq[:]))
        localization = extractor.get_subcellular_location(embedding)
        expected_localization = normalize_location(
            record.description.split(" ")[1][:-2]
        )
        actual_localization = normalize_location(str(localization.localization))
        results.append(expected_localization == actual_localization)

    actual_accuracy = numpy.asarray(results).mean()
    assert actual_accuracy == pytest.approx(expected_accuracy)

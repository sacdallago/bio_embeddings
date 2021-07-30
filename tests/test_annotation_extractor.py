import os
from typing import Type

import pytest

from bio_embeddings.embed import (
    EmbedderInterface,
    SeqVecEmbedder,
    ProtTransBertBFDEmbedder,
)
from bio_embeddings.extract import BasicAnnotationExtractor
from bio_embeddings.extract.annotations import Location, Membrane


@pytest.mark.skipif(os.environ.get("SKIP_SLOW_TESTS"), reason="This test is very slow")
@pytest.mark.parametrize(
    "model_type,embedder_class",
    [
        ("seqvec_from_publication", SeqVecEmbedder),
        ("bert_from_publication", ProtTransBertBFDEmbedder),
    ],
)
def test_basic_annotation_extractor(
    model_type: str, embedder_class: Type[EmbedderInterface]
):
    """Check that BasicAnnotationExtractor passes (without checking correctness)"""
    sequence = "SEQWENCE"
    embedding = embedder_class().embed(sequence)
    annotation_extractor = BasicAnnotationExtractor(model_type)
    annotations = annotation_extractor.get_annotations(embedding)
    # Check that results look reasonable
    assert len(annotations.DSSP3) == len(sequence)
    assert len(annotations.DSSP8) == len(sequence)
    assert len(annotations.disorder) == len(sequence)
    assert isinstance(annotations.localization, Location)
    assert isinstance(annotations.membrane, Membrane)

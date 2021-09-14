import os
from datetime import timedelta

import pytest
from hypothesis import given, settings
from hypothesis_bio import protein

from bio_embeddings.embed import EmbedderInterface, name_to_embedder
from tests.shared import check_embedding


# We need to create each embedder once for all hypothesis tests
# and destroy it before the next test so we don't run out of memory
# https://stackoverflow.com/a/44568273/3549270


@pytest.fixture(scope="session")
def cached_embedder(request) -> EmbedderInterface:
    return request.param()


@pytest.mark.skipif(
    not os.environ.get("RUN_VERY_SLOW_TESTS"),
    reason="Hypothesis is the slowest of them all",
)
@pytest.mark.parametrize("cached_embedder", name_to_embedder.values(), indirect=True)
@settings(deadline=timedelta(seconds=10))
@given(sequence=protein(uppercase_only=True))
def test_foo(sequence, cached_embedder: EmbedderInterface):
    embedding = cached_embedder.embed(sequence)
    check_embedding(cached_embedder, embedding, sequence)

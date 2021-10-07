from typing import Callable

import pytest
from pandas import read_csv

from bio_embeddings.extract.bindEmbed21 import BindEmbed21HBIAnnotationExtractor

# >Q9RN60
# groundtruth values for small molecule binding
Y = list("0000000000000-----------------------------------------------"
         "------------------------------------------------------------"
         "M-M---------------------------------------------------------"
         "-----------------------------M------------------------------"
         "----------------------00000")


@pytest.mark.parametrize(
    "get_extractor,expected_accuracy",
    [
        (
            lambda: BindEmbed21HBIAnnotationExtractor(),
            1.000,
        )
    ],
)
def test_binding_residue_annotation_extractor(
    get_extractor: Callable[[], BindEmbed21HBIAnnotationExtractor],
    expected_accuracy: float,
):
    """Check that BasicAnnotationExtractor passes (without checking correctness)"""
    extractor = get_extractor()

    hit = read_csv('test-data/bindEmbed21HBI/Q9RN60_hit.tsv', sep="\t")

    prediction = extractor.get_binding_residues(hit.iloc[0].to_dict())

    results = [
        actual.value == predicted
        for actual, predicted in zip(prediction.small_molecules, Y)
    ]

    actual_accuracy = round(sum(results) / len(results), 3)
    assert actual_accuracy == pytest.approx(expected_accuracy)

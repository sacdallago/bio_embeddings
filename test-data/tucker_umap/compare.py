"""Compares the csv written by running the pipeline with a reference for bert and tucker"""

from pathlib import Path

import pandas


def compare():
    root = Path(__file__).parent
    for embedding_type in ["bert", "tucker"]:
        expected = root.joinpath(f"merged_annotation_file_{embedding_type}.csv")
        expected = pandas.read_csv(expected)
        actual = root.joinpath(
            f"projected_tucker/visualize_{embedding_type}/merged_annotation_file.csv"
        )
        actual = pandas.read_csv(actual)
        pandas.testing.assert_frame_equal(expected, actual, check_exact=False)


if __name__ == "__main__":
    compare()

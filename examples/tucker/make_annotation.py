"""This script was used to generate tucker_annotations.csv. You shouldn't need to use it"""

from pathlib import Path

import pandas


def class_to_label(cath_class: int) -> str:
    """See http://www.cathdb.info/browse/tree"""
    mapping = {
        1: "Mainly Alpha",
        2: "Mainly Beta",
        3: "Alpha Beta",
        4: "Few secondary structures",
        6: "Special",
    }
    return mapping[cath_class]


def main():
    # Download this file from
    # http://download.cathdb.info/cath/releases/all-releases/v4_3_0/cath-classification-data/cath-domain-list-v4_3_0.txt
    mapping_df = pandas.read_fwf(
        "cath-domain-list-v4_3_0.txt",
        comment="#",
        colspecs=[(0, 7), (7, 13), (13, 19), (19, 25), (25, 31)],
        usecols=[0, 1, 2, 3, 4],
        names=["domain", "C", "A", "T", "H"],
    )

    fasta_file = "tucker_cath.fasta"
    ids = [i[1:] for i in Path(fasta_file).read_text().splitlines()[::2]]

    mapping = {
        domain: class_to_label(cath_class)
        for domain, cath_class in mapping_df[["domain", "C"]].itertuples(index=False)
    }
    records = [(i, mapping[i]) for i in ids]
    label_df = pandas.DataFrame.from_records(records, columns=["identifier", "label"])
    label_df.to_csv("cath_annotations_class.csv", index=False)


if __name__ == "__main__":
    main()

"""
BETA: in-silico mutagenesis using the substitution probabilities from ProtTrans-Bert-BFD
"""
from bio_embeddings.mutagenesis.constants import AMINO_ACIDS, PROBABILITIES_COLUMNS
from bio_embeddings.mutagenesis.pipeline import run, probabilities_as_dataframe

__all__ = [
    "AMINO_ACIDS",
    "PROBABILITIES_COLUMNS",
    "probabilities_as_dataframe",
    "run",
]

try:
    from bio_embeddings.mutagenesis.protbert_bfd import ProtTransBertBFDMutagenesis

    __all__.append("ProtTransBertBFDMutagenesis")
except ImportError:
    pass

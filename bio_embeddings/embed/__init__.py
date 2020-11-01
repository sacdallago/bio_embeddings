"""Language models to translate amino acid sequences into vector representations

All language models implement the :class:`EmbedderInterface`. You can embed a single
sequences with :meth:`EmbedderInterface.embed` or a list of
sequences with the :meth:`EmbedderInterface.embed_many` function. All except for CPCProt generate
per-residue embeddings, which you can summarize into a fixed size per-protein
embedding by calling :meth:`EmbedderInterface.reduce_per_protein`.
"""

import logging
from typing import Dict, Type

from bio_embeddings.embed.embedder_interfaces import EmbedderInterface

logger = logging.getLogger(__name__)

name_to_embedder: Dict[str, Type[EmbedderInterface]] = {}
__all__ = ["EmbedderInterface"]

# Transformers
try:
    from bio_embeddings.embed.protrans_albert_bfd_embedder import (
        ProtTransAlbertBFDEmbedder,
    )
    from bio_embeddings.embed.prottrans_bert_bfd_embedder import (
        ProtTransBertBFDEmbedder,
    )
    from bio_embeddings.embed.prottrans_xlnet_uniref100_embedder import (
        ProtTransXLNetUniRef100Embedder,
    )

    name_to_embedder[ProtTransAlbertBFDEmbedder.name] = ProtTransAlbertBFDEmbedder
    name_to_embedder[ProtTransBertBFDEmbedder.name] = ProtTransBertBFDEmbedder
    name_to_embedder[
        ProtTransXLNetUniRef100Embedder.name
    ] = ProtTransXLNetUniRef100Embedder

    __all__.append("ProtTransAlbertBFDEmbedder")
    __all__.append("ProtTransBertBFDEmbedder")
    __all__.append("ProtTransXLNetUniRef100Embedder")
except ImportError:
    logger.debug(
        "transformers extra not installed, Bert, Albert and XLNet will not be available"
    )

# Elmo / SeqVec
try:
    from bio_embeddings.embed.seqvec_embedder import SeqVecEmbedder

    name_to_embedder[SeqVecEmbedder.name] = SeqVecEmbedder
    __all__.append("SeqVecEmbedder")
except ImportError:
    logger.debug("allennlp extra not installed, SeqVec will not be available")

if not name_to_embedder:
    logger.warning(
        "No extra is installed, so none of the context dependent embedders are available! "
        "Please run `pip install bio-embeddings[all]`!"
    )

# ESM
try:
    from bio_embeddings.embed.esm_embedder import ESMEmbedder

    name_to_embedder[ESMEmbedder.name] = ESMEmbedder
    __all__.append("ESMEmbedder")
except ImportError:
    logger.debug("esm extra is not installed, ESM will not be available")

# UniRep
try:
    from bio_embeddings.embed.unirep_embedder import UniRepEmbedder

    name_to_embedder[UniRepEmbedder.name] = UniRepEmbedder
    __all__.append("UniRepEmbedder")
except ImportError:
    logger.debug("unirep extra not installedm UniRep will not be available")

# CPCProt
try:
    from bio_embeddings.embed.cpcprot_embedder import CPCProtEmbedder

    name_to_embedder[CPCProtEmbedder.name] = CPCProtEmbedder
    __all__.append("CPCProtEmbedder")
except ImportError:
    logger.debug("cpcprot extra not installed, CPCProt will not be available")

# PLUS
try:
    from bio_embeddings.embed.plus_rnn_embedder import PLUSRNNEmbedder
    name_to_embedder[PLUSRNNEmbedder.name] = PLUSRNNEmbedder
except ImportError:
    logger.debug("plus extra not installed, PLUSRNNEmbedder will not be available")

from bio_embeddings.embed.fasttext_embedder import FastTextEmbedder
from bio_embeddings.embed.glove_embedder import GloveEmbedder
from bio_embeddings.embed.word2vec_embedder import Word2VecEmbedder

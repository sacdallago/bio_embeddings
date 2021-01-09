"""Language models to translate amino acid sequences into vector representations

All language models implement the :class:`EmbedderInterface`. You can embed a single
sequences with :meth:`EmbedderInterface.embed` or a list of
sequences with the :meth:`EmbedderInterface.embed_many` function. All except
CPCProt and UniRep generate per-residue embeddings, which you can summarize
into a fixed size per-protein embedding by calling
:meth:`EmbedderInterface.reduce_per_protein`. CPCProt only generates a
per-protein embedding (``reduce_per_protein`` does nothing), while UniRep
includes a start token, so the embedding is one longer than the protein.

Instead of using ``bio_embeddings[all]``, it's possible to only install
some embedders by selecting specific extras:

* ``allennlp``: seqvec
* ``transformers``: prottrans_albert_bfd, prottrans_bert_bfd, protrans_xlnet_uniref100, prottrans_t5_bfd
* ``jax``-unirep: unirep
* ``esm``: esm
* ``cpcprot``: cpcprot
* ``plus``: plus_rnn
"""

import logging
from typing import Dict, Type

from bio_embeddings.embed.embedder_interfaces import EmbedderInterface

logger = logging.getLogger(__name__)

name_to_embedder: Dict[str, Type[EmbedderInterface]] = {}
__all__ = ["EmbedderInterface"]

# Transformers
try:
    from bio_embeddings.embed.prottrans_albert_bfd_embedder import (
        ProtTransAlbertBFDEmbedder,
    )
    from bio_embeddings.embed.prottrans_bert_bfd_embedder import (
        ProtTransBertBFDEmbedder,
    )
    from bio_embeddings.embed.prottrans_xlnet_uniref100_embedder import (
        ProtTransXLNetUniRef100Embedder,
    )
    from bio_embeddings.embed.prottrans_t5_bfd_embedder import ProtTransT5BFDEmbedder

    name_to_embedder[ProtTransAlbertBFDEmbedder.name] = ProtTransAlbertBFDEmbedder
    name_to_embedder[ProtTransBertBFDEmbedder.name] = ProtTransBertBFDEmbedder
    name_to_embedder[
        ProtTransXLNetUniRef100Embedder.name
    ] = ProtTransXLNetUniRef100Embedder
    name_to_embedder[ProtTransT5BFDEmbedder.name] = ProtTransT5BFDEmbedder

    __all__.append(ProtTransAlbertBFDEmbedder.__name__)
    __all__.append(ProtTransBertBFDEmbedder.__name__)
    __all__.append(ProtTransXLNetUniRef100Embedder.__name__)
    __all__.append(ProtTransT5BFDEmbedder.__name__)
except ImportError:
    logger.debug(
        "transformers extra not installed, Bert, Albert and XLNet will not be available"
    )

# Elmo / SeqVec
try:
    from bio_embeddings.embed.seqvec_embedder import SeqVecEmbedder

    name_to_embedder[SeqVecEmbedder.name] = SeqVecEmbedder
    __all__.append(SeqVecEmbedder.__name__)
except ImportError:
    logger.debug("allennlp extra not installed, SeqVec will not be available")

if not name_to_embedder:
    logger.warning(
        "No extra is installed, so none of the context dependent embedders are available! "
        "Please run `pip install bio-embeddings[all]`!"
    )

# ESM
try:
    from bio_embeddings.embed.esm_embedder import ESMEmbedder, ESM1bEmbedder

    name_to_embedder[ESMEmbedder.name] = ESMEmbedder
    name_to_embedder[ESM1bEmbedder.name] = ESM1bEmbedder
    __all__.append(ESMEmbedder.__name__)
    __all__.append(ESM1bEmbedder.__name__)
except ImportError:
    logger.debug("esm extra is not installed, ESM1 and ESM1b will not be available")

# UniRep
try:
    from bio_embeddings.embed.unirep_embedder import UniRepEmbedder

    name_to_embedder[UniRepEmbedder.name] = UniRepEmbedder
    __all__.append(UniRepEmbedder.__name__)
except ImportError:
    logger.debug("unirep extra not installedm UniRep will not be available")

# CPCProt
try:
    from bio_embeddings.embed.cpcprot_embedder import CPCProtEmbedder

    name_to_embedder[CPCProtEmbedder.name] = CPCProtEmbedder
    __all__.append(CPCProtEmbedder.__name__)
except ImportError:
    logger.debug("cpcprot extra not installed, CPCProt will not be available")

# PLUS
try:
    from bio_embeddings.embed.plus_rnn_embedder import PLUSRNNEmbedder

    name_to_embedder[PLUSRNNEmbedder.name] = PLUSRNNEmbedder
    __all__.append(PLUSRNNEmbedder.__name__)
except ImportError:
    logger.debug("plus extra not installed, PLUSRNNEmbedder will not be available")

# Bepler - should always work
try:
    from bio_embeddings.embed.bepler_embedder import BeplerEmbedder

    name_to_embedder[BeplerEmbedder.name] = BeplerEmbedder
    __all__.append(BeplerEmbedder.__name__)
except ImportError:
    logger.debug("bepler extra not installed, PLUSRNNEmbedder will not be available")

# Unmaintained embedders
from bio_embeddings.embed.fasttext_embedder import FastTextEmbedder
from bio_embeddings.embed.glove_embedder import GloveEmbedder
from bio_embeddings.embed.word2vec_embedder import Word2VecEmbedder

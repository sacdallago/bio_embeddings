import logging
from typing import Dict, Type

from bio_embeddings.embed.embedder_interfaces import EmbedderInterface

logger = logging.getLogger(__name__)

name_to_embedder: Dict[str, Type[EmbedderInterface]] = {}

# Transformers
try:
    from bio_embeddings.embed.protrans_albert_bfd_embedder import (
        ProtTransAlbertBFDEmbedder,
    )
    from bio_embeddings.embed.prottrans_bert_bfd_embedder import (
        ProtTransBertBFDEmbedder,
    )
    from bio_embeddings.embed.xlnet_embedder import ProtTransXLNetUniRef100Embedder

    name_to_embedder[ProtTransAlbertBFDEmbedder.name] = ProtTransAlbertBFDEmbedder
    name_to_embedder[ProtTransBertBFDEmbedder.name] = ProtTransBertBFDEmbedder
    name_to_embedder[
        ProtTransXLNetUniRef100Embedder.name
    ] = ProtTransXLNetUniRef100Embedder
except ImportError:
    logger.debug(
        "transformers extra not installed, Bert, Albert and XLNet will not be available"
    )

# Elmo / SeqVec
try:
    from bio_embeddings.embed.seqvec_embedder import SeqVecEmbedder

    name_to_embedder[SeqVecEmbedder.name] = SeqVecEmbedder
except ImportError:
    logger.debug("allennlp extra not installed, SeqVec will not be available")

if not name_to_embedder:
    logger.warning(
        "No extra is installed, so none of the context dependent embedders are available! "
        "Please run `pip install bio-embeddings[all]`!"
    )

# UniRep
try:
    from bio_embeddings.embed.unirep_embedder import UniRepEmbedder

    name_to_embedder[UniRepEmbedder.name] = UniRepEmbedder
except ImportError:
    logger.debug("unirep extra not installed and will not be available")

from bio_embeddings.embed.fasttext_embedder import FastTextEmbedder
from bio_embeddings.embed.glove_embedder import GloveEmbedder
from bio_embeddings.embed.word2vec_embedder import Word2VecEmbedder

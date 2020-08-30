import logging
from typing import Dict, Type

from bio_embeddings.embed.embedder_interfaces import EmbedderInterface

logger = logging.getLogger(__name__)

name_to_embedder: Dict[str, Type[EmbedderInterface]] = {}

# Transformers
try:
    from bio_embeddings.embed.albert_embedder import AlbertEmbedder
    from bio_embeddings.embed.bert_embedder import BertEmbedder
    from bio_embeddings.embed.xlnet_embedder import XLNetEmbedder

    name_to_embedder["albert"] = AlbertEmbedder
    name_to_embedder["bert"] = BertEmbedder
    name_to_embedder["xlnet"] = XLNetEmbedder
except ImportError:
    logger.debug(
        "transformers extra not installed, Bert, Albert and XLNet will not be available"
    )

# Elmo
try:
    from bio_embeddings.embed.seqvec_embedder import SeqVecEmbedder

    name_to_embedder["seqvec"] = SeqVecEmbedder
except ImportError:
    logger.debug("allennlp extra not installed, SeqVec will not be available")

if not name_to_embedder:
    logger.warning(
        "No extra is installed, so none of the context dependent embedders are available! "
        "Please run `pip install bio-embeddings[all]`!"
    )

# Unirep
try:
    from bio_embeddings.embed.unirep_embedder import UniRepEmbedder

    name_to_embedder["unirep"] = UniRepEmbedder
except ImportError:
    logger.debug("unirep extra not installed and will not be available")

from bio_embeddings.embed.fasttext_embedder import FastTextEmbedder
from bio_embeddings.embed.glove_embedder import GloveEmbedder
from bio_embeddings.embed.word2vec_embedder import Word2VecEmbedder

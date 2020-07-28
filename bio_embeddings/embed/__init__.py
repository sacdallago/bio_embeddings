from typing import Dict, Type

from bio_embeddings.embed.albert_embedder import AlbertEmbedder
from bio_embeddings.embed.bert_embedder import BertEmbedder
from bio_embeddings.embed.embedder_interfaces import EmbedderInterface
from bio_embeddings.embed.fasttext_embedder import FastTextEmbedder
from bio_embeddings.embed.glove_embedder import GloveEmbedder
from bio_embeddings.embed.seqvec_embedder import SeqVecEmbedder
from bio_embeddings.embed.word2vec_embedder import Word2VecEmbedder
from bio_embeddings.embed.xlnet_embedder import XLNetEmbedder

name_to_embedder: Dict[str, Type[EmbedderInterface]] = {
    "seqvec": SeqVecEmbedder,
    "albert": AlbertEmbedder,
    "bert": BertEmbedder,
    "xlnet": XLNetEmbedder,
}

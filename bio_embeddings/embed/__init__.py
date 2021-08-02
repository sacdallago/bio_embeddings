"""Language models to translate amino acid sequences into vector representations

All language models implement the :class:`EmbedderInterface`. You can embed a single
sequences with :meth:`EmbedderInterface.embed` or a list of
sequences with the :meth:`EmbedderInterface.embed_many` function. All except
CPCProt and UniRep generate per-residue embeddings, which you can summarize
into a fixed size per-protein embedding by calling
:meth:`EmbedderInterface.reduce_per_protein`. CPCProt only generates a
per-protein embedding (``reduce_per_protein`` does nothing), while UniRep
includes a start token, so the embedding is one longer than the protein.
UniRep, GloVe, fastText and word2vec only support the CPU.

:class:`OneHotEncodingEmbedder` offers a naive baseline to compare the language model
embeddings against, with one hot encoding as per-residue and amino acid composition
as per-protein embedding. It accepts keyword arguments but ignores them since it does
not do any notable computation.

Instead of using ``bio_embeddings[all]``, it's possible to only install
some embedders by selecting specific extras:

* ``allennlp``: seqvec
* ``transformers``: prottrans_albert_bfd, prottrans_bert_bfd, protrans_xlnet_uniref100, prottrans_t5_bfd, prottrans_t5_uniref50, prottrans_t5_xl_u50
* ``jax``-unirep: unirep
* ``esm``: esm
* ``cpcprot``: cpcprot
* ``plus``: plus_rnn

Model sizes
-----------

The disk size represents the size of the unzipped models or the combination of all files necessary for a particular
embeddder. The GPU and CPU sizes are only for loading the model into GPU memory (VRAM) or the main RAM without the
memory required to do any computation. They were measured for one specific set of hardware and software
(Quadro RTX 8000, CUDA 11.1, torch 1.7.1, x86 64-bit Ubuntu 18.04) and will vary for different setups.

==============================================  ==============  =============   ==============
Model                                           Disk size (GB)  GPU size (GB)   CPU size (GB)
==============================================  ==============  =============   ==============
bepler                                          0.1             1.4             0.2
bert_from_publication                           0.008           1.1             0.006
cpcprot                                         0.007           1.1             0.01
deepblast                                       0.4             1.4             0.26
esm                                             6.3             3.9             2.7
esm1b                                           7.3             3.8             2.6
esm1v                                           7.3             3.9             2.6
fasttext                                        0.05            n/a             0.03
glove                                           0.06            n/a             0.03
one_hot_encoding                                n/a             n/a             n/a
pb_tucker                                       0.009           1.0             0.02
plus_rnn                                        0.06            1.2             0.1
prottrans_albert_bfd                            0.9             2.0             1.8
prottrans_bert_bfd                              1.6             2.8             3.4
prottrans_t5_bfd                                7.2             5.9             16.1
prottrans_t5_uniref50                           7.2             5.9             16.1
prottrans_t5_xl_u50                             7.2             5.9             16.1
prottrans_xlnet_uniref100                       1.6             2.7             3.3
seqvec                                          0.4             1.6             0.5
seqvec_from_publication                         0.004           1.1             0.006
unirep                                          n/a             n/a             0.2
word2vec                                        0.07            n/a             0.06
==============================================  ==============  =============   ==============
"""

import logging
from typing import Dict, Type

from bio_embeddings.embed.embedder_interfaces import EmbedderInterface
from bio_embeddings.embed.fasttext_embedder import FastTextEmbedder
from bio_embeddings.embed.glove_embedder import GloveEmbedder
from bio_embeddings.embed.one_hot_encoding_embedder import OneHotEncodingEmbedder
from bio_embeddings.embed.word2vec_embedder import Word2VecEmbedder

logger = logging.getLogger(__name__)

name_to_embedder: Dict[str, Type[EmbedderInterface]] = {
    OneHotEncodingEmbedder.name: OneHotEncodingEmbedder,
    FastTextEmbedder.name: FastTextEmbedder,
    GloveEmbedder.name: GloveEmbedder,
    Word2VecEmbedder.name: Word2VecEmbedder,
}

__all__ = [
    "EmbedderInterface",
    "OneHotEncodingEmbedder",
    "FastTextEmbedder",
    "GloveEmbedder",
    "Word2VecEmbedder",
]

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
    from bio_embeddings.embed.prottrans_t5_embedder import (
        ProtTransT5BFDEmbedder,
        ProtTransT5UniRef50Embedder,
        ProtTransT5XLU50Embedder,
    )

    name_to_embedder[ProtTransAlbertBFDEmbedder.name] = ProtTransAlbertBFDEmbedder
    name_to_embedder[ProtTransBertBFDEmbedder.name] = ProtTransBertBFDEmbedder
    name_to_embedder[
        ProtTransXLNetUniRef100Embedder.name
    ] = ProtTransXLNetUniRef100Embedder
    name_to_embedder[ProtTransT5BFDEmbedder.name] = ProtTransT5BFDEmbedder
    name_to_embedder[ProtTransT5UniRef50Embedder.name] = ProtTransT5UniRef50Embedder
    name_to_embedder[ProtTransT5XLU50Embedder.name] = ProtTransT5XLU50Embedder

    __all__.append(ProtTransAlbertBFDEmbedder.__name__)
    __all__.append(ProtTransBertBFDEmbedder.__name__)
    __all__.append(ProtTransXLNetUniRef100Embedder.__name__)
    __all__.append(ProtTransT5BFDEmbedder.__name__)
    __all__.append(ProtTransT5UniRef50Embedder.__name__)
    __all__.append(ProtTransT5XLU50Embedder.__name__)
except ModuleNotFoundError as e:
    # Check that the error is actually the missing transformers extra
    if e.name != "transformers":
        raise
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
    from bio_embeddings.embed.esm_embedder import ESMEmbedder, ESM1bEmbedder, ESM1vEmbedder

    name_to_embedder[ESMEmbedder.name] = ESMEmbedder
    name_to_embedder[ESM1bEmbedder.name] = ESM1bEmbedder
    name_to_embedder[ESM1vEmbedder.name] = ESM1vEmbedder
    __all__.append(ESMEmbedder.__name__)
    __all__.append(ESM1bEmbedder.__name__)
    __all__.append(ESM1vEmbedder.__name__)
except ImportError:
    logger.debug("esm extra is not installed, ESM1, ESM1b and ESM1v will not be available")

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

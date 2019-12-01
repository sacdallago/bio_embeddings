import tempfile
from copy import deepcopy
from bio_embeddings.embed import (
    SeqVecEmbedder as _SeqVecEmbedder,
    Word2VecEmbedder as _Word2VecEmbedder,
    FastTextEmbedder as _FastTextEmbedder,
    GloveEmbedder as _GloveEmbedder,
    TransformerXLEmbedder as _TransformerXLEmbedder
)
from bio_embeddings.utilities import get_model_file


# To make it easier to final users of the embeddings as a package,
# auto-download missing files!


def SeqVecEmbedder(**kwargs) -> _SeqVecEmbedder:
    necessary_files = ['weights_file', 'options_file']

    if kwargs.get('seqvec_version') == 2 or kwargs.get('vocabulary_file'):
        necessary_files.append('vocabulary_file')
        kwargs['seqvec_version'] = 2

    for file in necessary_files:
        if not kwargs.get(file):
            f = tempfile.NamedTemporaryFile()

            get_model_file(
                model='seqvecv{}'.format(str(kwargs['seqvec_version'])),
                file=file,
                path=f.name
            )

            kwargs[file] = f.name

    return _SeqVecEmbedder(**kwargs)


def Word2VecEmbedder(**kwargs) -> _Word2VecEmbedder:
    necessary_files = ['model_file']

    for file in necessary_files:
        if not kwargs.get(file):
            f = tempfile.NamedTemporaryFile()

            get_model_file(
                model='word2vec',
                file=file,
                path=f.name
            )

            kwargs[file] = f.name

    return _Word2VecEmbedder(**kwargs)

def FastTextEmbedder(**kwargs) -> _FastTextEmbedder:
    necessary_files = ['model_file']

    for file in necessary_files:
        if not kwargs.get(file):
            f = tempfile.NamedTemporaryFile()

            get_model_file(
                model='fasttext',
                file=file,
                path=f.name
            )

            kwargs[file] = f.name

    return _FastTextEmbedder(**kwargs)


def GloveEmbedder(**kwargs) -> _GloveEmbedder:
    necessary_files = ['model_file']

    for file in necessary_files:
        if not kwargs.get(file):
            f = tempfile.NamedTemporaryFile()

            get_model_file(
                model='glove',
                file=file,
                path=f.name
            )

            kwargs[file] = f.name

    return _GloveEmbedder(**kwargs)


def TransformerXLEmbedder(**kwargs) -> _TransformerXLEmbedder:
    necessary_files = ['model_file', 'vocabulary_file']

    model_size = kwargs.get('model', 'base')

    for file in necessary_files:
        if not kwargs.get(file):
            f = tempfile.NamedTemporaryFile()

            get_model_file(
                model='transformerxl_{}'.format(model_size),
                file=file,
                path=f.name
            )

            kwargs[file] = f.name

    return _TransformerXLEmbedder(**kwargs)

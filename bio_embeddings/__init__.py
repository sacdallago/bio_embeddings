import tempfile
from bio_embeddings.embed import (
    SeqVecEmbedder as _SeqVecEmbedder,
    Word2VecEmbedder as _Word2VecEmbedder,
    FastTextEmbedder as _FastTextEmbedder,
    GloveEmbedder as _GloveEmbedder,
    TransformerXLEmbedder as _TransformerXLEmbedder,
    AlbertEmbedder as _AlbertEmbedder
)

from bio_embeddings.utilities import get_model_file, get_model_directories_from_zip

_temporary_files = list()

# To make it easier for end-users of the embeddings as a package,
# auto-download missing files!


class SeqVecEmbedder(_SeqVecEmbedder):

    def __init__(self, **kwargs):
        necessary_files = ['weights_file', 'options_file']

        if kwargs.get('seqvec_version') == 2 or kwargs.get('vocabulary_file'):
            necessary_files.append('vocabulary_file')
            kwargs['seqvec_version'] = 2

        for file in necessary_files:
            if not kwargs.get(file):
                f = tempfile.NamedTemporaryFile()
                _temporary_files.append(f)

                get_model_file(
                    model='seqvecv{}'.format(str(kwargs.get('seqvec_version', 1))),
                    file=file,
                    path=f.name
                )

                kwargs[file] = f.name
        super().__init__(**kwargs)


class AlbertEmbedder(_AlbertEmbedder):

    def __init__(self, **kwargs):
        necessary_directories = ['model_directory']

        for directory in necessary_directories:
            if not kwargs.get(directory):
                f = tempfile.mkdtemp()
                _temporary_files.append(f)

                get_model_directories_from_zip(
                    model='albert',
                    directory=directory,
                    path=f
                )

                kwargs[directory] = f
        super().__init__(**kwargs)


class Word2VecEmbedder(_Word2VecEmbedder):
    def __init__(self, **kwargs):
        necessary_files = ['model_file']

        for file in necessary_files:
            if not kwargs.get(file):
                f = tempfile.NamedTemporaryFile()
                _temporary_files.append(f)

                get_model_file(
                    model='word2vec',
                    file=file,
                    path=f.name
                )

                kwargs[file] = f.name

        super().__init__(**kwargs)


class FastTextEmbedder(_FastTextEmbedder):
    def __init__(self, **kwargs):
        necessary_files = ['model_file']

        for file in necessary_files:
            if not kwargs.get(file):
                f = tempfile.NamedTemporaryFile()
                _temporary_files.append(f)

                get_model_file(
                    model='fasttext',
                    file=file,
                    path=f.name
                )

                kwargs[file] = f.name

        super().__init__(**kwargs)


class GloveEmbedder(_GloveEmbedder):
    def __init__(self, **kwargs):
        necessary_files = ['model_file']

        for file in necessary_files:
            if not kwargs.get(file):
                f = tempfile.NamedTemporaryFile()
                _temporary_files.append(f)

                get_model_file(
                    model='glove',
                    file=file,
                    path=f.name
                )

                kwargs[file] = f.name

        super().__init__(**kwargs)


class TransformerXLEmbedder(_TransformerXLEmbedder):
    def __init__(self, **kwargs):
        necessary_files = ['model_file', 'vocabulary_file']

        model_size = kwargs.get('model', 'base')

        for file in necessary_files:
            if not kwargs.get(file):
                f = tempfile.NamedTemporaryFile()
                _temporary_files.append(f)

                get_model_file(
                    model='transformerxl_{}'.format(model_size),
                    file=file,
                    path=f.name
                )

                kwargs[file] = f.name

        super().__init__(**kwargs)

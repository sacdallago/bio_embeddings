from io import StringIO
from os import path
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

import h5py
from Bio import SeqIO, SeqRecord

from bio_embeddings.embed import SeqVecEmbedder, AlbertEmbedder
from webserver.database import get_file, write_file
from webserver.tasks import task_keeper

_model_dir = path.join(Path(path.abspath(__file__)).parent.parent.parent, 'models')

# SeqVec
_seqvec_weight_file = path.join(_model_dir, 'elmov1', 'weights.hdf5')
_seqvec_options_file = path.join(_model_dir, 'elmov1', 'options.json')

# Albert
_albert_model_dir = path.join(_model_dir, 'albert')


def _get_embeddings(protein_generator: Iterable[SeqRecord], embedder, reduced_embeddings_file):
    max_amino_acids = 15000
    protein_data = [(entry.id, str(entry.seq)) for entry in protein_generator]
    ids, sequences = zip(*protein_data)

    for sequence_id, embedding in zip(ids, embedder.embed_many(sequences, max_amino_acids)):
        reduced_embeddings_file.create_dataset(
            sequence_id,
            data=embedder.reduce_per_protein(embedding)
        )


@task_keeper.task()
def get_embeddings(job_identifier, embedder='seqvec'):
    if embedder == 'seqvec':
        embedder = SeqVecEmbedder(weights_file=_seqvec_weight_file, options_file=_seqvec_options_file)
    elif embedder == 'albert':
        embedder = AlbertEmbedder(model_directory=_albert_model_dir)

    with get_file(job_identifier, "sequences_file") as db_file:
        file_content = StringIO(db_file.read().decode("utf-8"))
        protein_generator: Iterable[SeqRecord] = SeqIO.parse(file_content, 'fasta')
        db_file.close()

    with NamedTemporaryFile() as temp_file:
        with h5py.File(temp_file.name, "w") as reduced_embeddings_file:
            _get_embeddings(protein_generator, embedder, reduced_embeddings_file)

        write_file(job_identifier, "reduced_embeddings_file", temp_file.name)

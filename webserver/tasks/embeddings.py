import h5py
from io import StringIO
from os import path
from tempfile import NamedTemporaryFile
from pathlib import Path
from bio_embeddings.utilities import read_fasta_file_generator, read_fasta_file
from bio_embeddings.embed import SeqVecEmbedder, AlbertEmbedder
from webserver.database import get_file, write_file
from webserver.tasks import task_keeper

_model_dir = path.join(Path(path.abspath(__file__)).parent.parent.parent, 'models')

# SeqVec
_seqvec_weight_file = path.join(_model_dir, 'elmov1', 'weights.hdf5')
_seqvec_options_file = path.join(_model_dir, 'elmov1', 'options.json')

# Albert
_albert_model_dir = path.join(_model_dir, 'albert')


def _get_embeddings(protein_generator, embedder, reduced_embeddings_file):
    # Embed iteratively (5k sequences at the time)
    max_amino_acids_RAM = 15000

    candidates = list()
    aa_count = 0

    for sequence in protein_generator:
        candidates.append(sequence)
        aa_count += len(sequence)

        if aa_count + len(sequence) > max_amino_acids_RAM:
            embeddings = embedder.embed_many([protein.seq for protein in candidates])

            for index, protein in enumerate(candidates):
                reduced_embeddings_file.create_dataset(
                    protein.id,
                    data=embedder.reduce_per_protein(embeddings[index])
                )

            # Reset
            aa_count = 0
            candidates = list()

    if candidates:
        embeddings = embedder.embed_many([protein.seq for protein in candidates])

        for index, protein in enumerate(candidates):
            reduced_embeddings_file.create_dataset(
                protein.id,
                data=embedder.reduce_per_protein(embeddings[index])
            )


@task_keeper.task()
def get_embeddings(job_identifier, embedder='seqvec'):
    if embedder == 'seqvec':
        embedder = SeqVecEmbedder(weights_file=_seqvec_weight_file, options_file=_seqvec_options_file)
    elif embedder == 'albert':
        embedder = AlbertEmbedder(model_directory=_albert_model_dir)

    with get_file(job_identifier, "sequences_file") as db_file:
        file_content = StringIO(db_file.read().decode("utf-8"))
        protein_generator = read_fasta_file(file_content)
        db_file.close()

    with NamedTemporaryFile() as temp_file:
        with h5py.File(temp_file.name, "w") as reduced_embeddings_file:
            _get_embeddings(protein_generator, embedder, reduced_embeddings_file)

        write_file(job_identifier, "reduced_embeddings_file", temp_file.name)

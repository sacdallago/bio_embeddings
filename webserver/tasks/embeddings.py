import h5py
from os import path
from tempfile import NamedTemporaryFile
from pathlib import Path
from bio_embeddings.utilities import read_fasta_file_generator
from bio_embeddings.embed import SeqVecEmbedder, AlbertEmbedder
from webserver.database import get_file, write_file
from webserver.tasks import task_keeper

_model_dir = path.join(Path(path.abspath(__file__)).parent.parent.parent, 'models')

# SeqVec
_seqvec_weight_file = path.join(_model_dir, 'elmov1', 'weights.hdf5')
_seqvec_options_file = path.join(_model_dir, 'elmov1', 'options.json')

# Albert
_albert_model_dir = path.join(_model_dir, 'albert')


def _get_embeddings(protein_generator, embedder, reduced_embeddings_file_path):
    # Embed iteratively (5k sequences at the time)
    max_amino_acids_RAM = 15000
    reduced_embeddings_file = h5py.File(reduced_embeddings_file_path, "w")

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

    # Close embeddings files
    reduced_embeddings_file.close()


@task_keeper.task()
def get_embeddings(job_identifier, embedder='seqvec'):
    if embedder == 'seqvec':
        embedder = SeqVecEmbedder(weights_file=_seqvec_weight_file, options_file=_seqvec_options_file)
    elif embedder == 'albert':
        embedder = AlbertEmbedder(model_directory=_albert_model_dir)

    temp_sequence_file = NamedTemporaryFile()
    with get_file(job_identifier, "sequences_file") as db_file:
        temp_sequence_file.write(db_file.read())
        db_file.close()

    protein_generator = read_fasta_file_generator(temp_sequence_file.name)

    with NamedTemporaryFile() as temp_file:
        _get_embeddings(protein_generator, embedder, temp_file)
        write_file(job_identifier, "reduced_embeddings_file", temp_file.name)

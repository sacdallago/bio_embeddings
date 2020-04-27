from os import path
from pathlib import Path
from webserver.tasks import task_keeper, IN_CELERY_WOKER_PROCESS
from celery.exceptions import SoftTimeLimitExceeded
from bio_embeddings.embed import SeqVecEmbedder


_model_dir = path.join(Path(path.abspath(__file__)).parent.parent.parent, 'models')

# SeqVec
_weight_file = path.join(_model_dir, 'elmov1', 'weights.hdf5')
_options_file = path.join(_model_dir, 'elmov1', 'options.json')

models = {}


def load_models():
    models['seqvec'] = SeqVecEmbedder(weights_file=_weight_file, options_file=_options_file)


# Only initialize the model if I'm a celery worker, otherwise it's just wasted RAM
if IN_CELERY_WOKER_PROCESS:
    print("Loading model...")
    load_models()


@task_keeper.task(time_limit=60*5, soft_time_limit=60*5, expires=60*60)
def get_embeddings(sequence, embedder='seqvec'):
    try:
        embedding = models.get(embedder, models['seqvec']).embed(sequence)

        return embedding
    except SoftTimeLimitExceeded:
        raise Exception("Time limit exceeded")

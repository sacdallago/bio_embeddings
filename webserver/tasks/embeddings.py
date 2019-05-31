from os import path
from pathlib import Path
from webserver.tasks import task_keeper, IN_CELERY_WOKER_PROCESS
from celery.exceptions import SoftTimeLimitExceeded
from bio_embeddings.embedders import ElmoEmbedder


_model_dir = path.join(Path(path.abspath(__file__)).parent.parent.parent, 'models')

# Elmo V1
_weight_file = path.join(_model_dir, 'elmov1', 'weights.hdf5')
_options_file = path.join(_model_dir, 'elmov1', 'options.json')
_secondary_structure_checkpoint_file = path.join(_model_dir, 'elmov1', 'secstruct_checkpoint.pt')
_subcellular_location_checkpoint_file = path.join(_model_dir, 'elmov1', 'subcell_checkpoint.pt')


models = {}


def load_models():

    models['elmo'] = ElmoEmbedder(weights_file=_weight_file,
                                  options_file=_options_file,
                                  secondary_structure_checkpoint_file=_secondary_structure_checkpoint_file,
                                  subcellular_location_checkpoint_file=_subcellular_location_checkpoint_file
                                  )


# Only initialize the model if I'm a celery worker, otherwise it's just wasted RAM
if IN_CELERY_WOKER_PROCESS:
    print("Loading model...")
    load_models()


@task_keeper.task(time_limit=60*5, soft_time_limit=60*5, expires=60*60)
def get_embedding(sequence, embedder='elmo'):
    try:
        embedding = models.get(embedder, models['elmo']).embed(sequence)

        return embedding
    except SoftTimeLimitExceeded:
        raise Exception("Time limit exceeded")


@task_keeper.task(time_limit=60*5, soft_time_limit=60*5, expires=60*60)
def get_features(sequence, embedder='elmo'):
    try:
        model = models.get(embedder, models['elmo'])

        model.embed(sequence)
        features = model.get_features()

        return features.to_dict()
    except SoftTimeLimitExceeded:
        raise Exception("Time limit exceeded")

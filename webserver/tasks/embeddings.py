from os import path
from pathlib import Path
from webserver.tasks import task_keeper, IN_CELERY_WOKER_PROCESS
from celery.exceptions import SoftTimeLimitExceeded
from bio_embeddings.embedders import ElmoEmbedder


_model_dir = path.join(Path(path.abspath(__file__)).parent.parent, 'models')

# Elmo V1
_weight_file = path.join(_model_dir, 'weights.hdf5')
_options_file = path.join(_model_dir, 'options.json')
_secondary_structure_checkpoint_file = path.join(_model_dir, 'sec_struct.chk')
_subcellular_location_checkpoint_file = path.join(_model_dir, 'sub_loc.chk')


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
def get_seqvec(sequence, embedder='elmo'):
    try:
        embedding = models.get(embedder, models['elmo']).embed(sequence)

        return embedding
    except SoftTimeLimitExceeded:
        raise Exception("Time limit exceeded")


def get_features(embedder='elmo'):
    models.get(embedder, models['elmo']).features().to_dict()

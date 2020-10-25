from webserver.tasks import task_keeper
from webserver.utilities.configuration import configuration

model = None
featureExtractor = None

if configuration['celery']['celery_worker_type'] == "seqvec":
    from bio_embeddings.embed import SeqVecEmbedder
    from bio_embeddings.extract.basic.BasicAnnotationExtractor import BasicAnnotationExtractor

    model = SeqVecEmbedder(
        options_file=configuration['seqvec']['options_file'],
        weights_file=configuration['seqvec']['weights_file']
    )

    featureExtractor = BasicAnnotationExtractor("seqvec_from_publication")


@task_keeper.task()
def get_seqvec_embeddings_sync(sequence: str):
    return model.embed(sequence)


@task_keeper.task()
def get_seqvec_annotations_sync(sequence: str):
    embedding = model.embed(sequence)
    annotations = featureExtractor.get_annotations(embedding)

    return annotations

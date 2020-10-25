from webserver.tasks import task_keeper
from webserver.utilities.configuration import configuration

model = None
featureExtractor = None

if configuration['celery']['celery_worker_type'] == "protbert":
    from bio_embeddings.embed import ProtTransBertBFDEmbedder
    from bio_embeddings.extract.basic.BasicAnnotationExtractor import BasicAnnotationExtractor

    model = ProtTransBertBFDEmbedder(
        model_directory=configuration['prottrans_bert_bfd']['model_directory']
    )

    featureExtractor = BasicAnnotationExtractor("bert_from_publication")


@task_keeper.task()
def get_protbert_embeddings_sync(sequence: str):
    return model.embed(sequence)


@task_keeper.task()
def get_protbert_annotations_sync(sequence: str):
    embedding = model.embed(sequence)
    annotations = featureExtractor.get_annotations(embedding)

    return annotations

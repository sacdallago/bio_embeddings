import os.path
from os import environ

model_directory = environ["MODEL_DIRECTORY"]

configuration = {
    # Web Stuff
    "web": {
        "max_amino_acids": int(environ.get("WEB_MAX_AMINO_ACIDS", 15000)),
        # Max content length determines the maximum size of files to be uploaded: 16 * 1024 * 1024 = 16MB
        "max_content_length": int(
            eval(environ.get("MAX_CONTENT_LENGTH", "16 * 1024 * 1024"))
        ),
        "mongo_url": environ.get("MONGO_URL", "mongodb://localhost:27017"),
        "celery_broker_url": environ["CELERY_BROKER_URL"],
    },
    # Bert stuff
    "prottrans_bert_bfd": {
        "model_directory": os.path.join(model_directory, "bert", "model_directory"),
        "max_amino_acids": int(environ.get("BERT_MAX_AMINO_ACIDS", 8000)),
        "secondary_structure_checkpoint_file": os.path.join(
            model_directory, "bert_from_publication_annotations_extractors/secondary_structure_checkpoint_file"
        ),
        "subcellular_location_checkpoint_file": os.path.join(
            model_directory, "bert/subcellular_location_checkpoint_file"
        ),
    },
    # SeqVec stuff
    "seqvec": {
        "weights_file": os.path.join(model_directory, "seqvec/weights_file"),
        "options_file": os.path.join(model_directory, "seqvec/options_file"),
        "max_amino_acids": int(environ.get("SEQVEC_MAX_AMINO_ACIDS", 20000)),
        "secondary_structure_checkpoint_file": os.path.join(
            model_directory, "seqvec_from_publication_annotations_extractors/secondary_structure_checkpoint_file"
        ),
        "subcellular_location_checkpoint_file": os.path.join(
            model_directory, "seqvec_from_publication_annotations_extractors/subcellular_location_checkpoint_file"
        ),
    },
}

import os.path
from os import environ

import sentry_sdk
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.flask import FlaskIntegration

if "SENTRY_DSN" in environ:
    sentry_sdk.init(
        dsn=environ["SENTRY_DSN"],
        integrations=[FlaskIntegration(), CeleryIntegration()],
    )

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
        "model_directory": os.path.join(model_directory, "prottrans_bert_bfd", "model_directory"),
        "max_amino_acids": int(environ.get("BERT_MAX_AMINO_ACIDS", 8000)),
        "secondary_structure_checkpoint_file": os.path.join(
            model_directory, "bert_from_publication_annotations_extractors", "secondary_structure_checkpoint_file"
        ),
        "subcellular_location_checkpoint_file": os.path.join(
            model_directory, "bert_from_publication_annotations_extractors", "subcellular_location_checkpoint_file"
        ),
        "la_subcellular_location_checkpoint_file": os.path.join(
            model_directory, "light_attention", "la_protbert_subcellular_location"
        ),
        "la_solubility_checkpoint_file": os.path.join(
            model_directory, "light_attention", "la_protbert_solubility"
        ),
        # TODO: add goPredSim stuff
    },
    # T5 stuff
    "prottrans_t5_xl_u50": {
        "model_directory": os.path.join(model_directory, "prottrans_t5_xl_u50", "model_directory"),
        "max_amino_acids": int(environ.get("T5_MAX_AMINO_ACIDS", 2000)),
        "secondary_structure_checkpoint_file": os.path.join(
            model_directory, "t5_xl_u50_from_publication_annotations_extractors", "secondary_structure_checkpoint_file"
        ),
        "subcellular_location_checkpoint_file": os.path.join(
            model_directory, "t5_xl_u50_from_publication_annotations_extractors", "subcellular_location_checkpoint_file"
        ),
        "la_subcellular_location_checkpoint_file": os.path.join(
            model_directory, "light_attention", "la_prott5_subcellular_location"
        ),
        "la_solubility_checkpoint_file": os.path.join(
            model_directory, "light_attention", "la_prott5_solubility"
        ),
    },
    # SeqVec stuff
    "seqvec": {
        "weights_file": os.path.join(model_directory, "seqvec", "weights_file"),
        "options_file": os.path.join(model_directory, "seqvec", "options_file"),
        "max_amino_acids": int(environ.get("SEQVEC_MAX_AMINO_ACIDS", 20000)),
        "secondary_structure_checkpoint_file": os.path.join(
            model_directory, "seqvec_from_publication_annotations_extractors", "secondary_structure_checkpoint_file"
        ),
        "subcellular_location_checkpoint_file": os.path.join(
            model_directory, "seqvec_from_publication_annotations_extractors", "subcellular_location_checkpoint_file"
        ),
        "go_reference_embeddings": os.path.join(
            model_directory, "goa", "seqvec_l1_embeddings.h5"
        ),
    },
    "goa": {
        "bpo": os.path.join(
            model_directory, "goa", "goa_annotations_2020_bpo.txt",
        ),
        "cco": os.path.join(
            model_directory, "goa", "goa_annotations_2020_cco.txt",
        ),
        "mfo": os.path.join(
            model_directory, "goa", "goa_annotations_2020_mfo.txt",
        ),
    },
    # Celery worker type
    "celery": {
        # Types can be, separated by comma:
        #  - nothing ==> pipeline async worker
        #  - pipeline ==> pipeline async worker
        #  - seqvec ==> takes sequences and returns embeddings (sync)
        #  - seqvec_annotations ==> takes embeddings and returns annotations (sync)
        #  - protbert ==> takes sequences and returns embeddings (sync)
        #  - protbert_annotations ==> takes embeddings and returns annotations (sync)

        "celery_worker_type": environ["CELERY_WORKER_TYPE"].split(",") if "CELERY_WORKER_TYPE" in environ else [],
    }
}

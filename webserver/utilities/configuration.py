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
    # T5 stuff
    "prottrans_t5_xl_u50": {
        "half_model_directory": os.path.join(model_directory, "prottrans_t5_xl_u50", "half_model_directory"),
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
    # BindEmbed21 stuff
    "bindembed21": {
        "model_1_file": os.path.join(model_directory, "bindembed21", "checkpoint1.pt"),
        "model_2_file": os.path.join(model_directory, "bindembed21", "checkpoint2.pt"),
        "model_3_file": os.path.join(model_directory, "bindembed21", "checkpoint3.pt"),
        "model_4_file": os.path.join(model_directory, "bindembed21", "checkpoint4.pt"),
        "model_5_file": os.path.join(model_directory, "bindembed21", "checkpoint5.pt")
    },
    "goa": {
        "bpo": os.path.join(
            model_directory, "goa", "goa_annotations_2022_bpo.txt",
        ),
        "cco": os.path.join(
            model_directory, "goa", "goa_annotations_2022_cco.txt",
        ),
        "mfo": os.path.join(
            model_directory, "goa", "goa_annotations_2022_mfo.txt",
        ),
        "go_reference_embeddings": os.path.join(
            model_directory, "goa", "prott5_reference_embeddings.h5"
        ),
    },
    # Colabfold stuff
    "colabfold": {
        "data_dir": os.path.join(model_directory, "colabfold")
    },
    # tembed
    "tmbed": {
        "model_0_file": os.path.join(model_directory, "tmbed", "cv_0.pt"),
        "model_1_file": os.path.join(model_directory, "tmbed", "cv_1.pt"),
        "model_2_file": os.path.join(model_directory, "tmbed", "cv_2.pt"),
        "model_3_file": os.path.join(model_directory, "tmbed", "cv_3.pt"),
        "model_4_file": os.path.join(model_directory, "tmbed", "cv_4.pt")
    },
    # Celery worker type
    "celery": {
        # Types can be, separated by comma:
        #  - nothing ==> pipeline async worker
        #  - pipeline ==> pipeline async worker
        #  - prott5 ==> takes sequences and returns embeddings (sync)
        #  - prott5_annotations ==> takes embeddings and returns annotations (sync)
        #  - colabfold ==> takes sequence and returns structure

        "celery_worker_type": environ["CELERY_WORKER_TYPE"].split(",") if "CELERY_WORKER_TYPE" in environ else [],
    }
}

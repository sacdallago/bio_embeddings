from os import environ

configuration = {
    # Web Stuff
    "web": {
        "max_amino_acids": int(environ.get('WEB_MAX_AMINO_ACIDS', 15000)),

        # Max content length determines the maximum size of files to be uploaded: 16 * 1024 * 1024 = 16MB
        "max_content_length": int(eval(environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))),
        "mongo_url": environ.get('MONGO_URL', "mongodb://localhost:27017")
    },

    # Bert stuff
    "bert": {
        "model_directory": environ['BERT_MODEL_DIRECTORY'],
        "max_amino_acids": int(environ.get('BERT_MAX_AMINO_ACIDS', 8000)),
        "secondary_structure_checkpoint_file": environ['BERT_SECONDARY_STRUCTURE_CHECKPOINT_FILE'],
        "subcellular_location_checkpoint_file": environ['BERT_SUBCELLULAR_LOCATION_CHECKPOINT_FILE']
    },

    # SeqVec stuff
    "seqvec": {
        "weights_file": environ['SEQVEC_WEIGHTS_FILE'],
        "options_file": environ['SEQVEC_OPTIONS_FILE'],
        "max_amino_acids": int(environ.get('SEQVEC_MAX_AMINO_ACIDS', 20000)),
        "secondary_structure_checkpoint_file": environ['SEQVEC_SECONDARY_STRUCTURE_CHECKPOINT_FILE'],
        "subcellular_location_checkpoint_file": environ['SEQVEC_SUBCELLULAR_LOCATION_CHECKPOINT_FILE']
    },
}

# The bio_embeddings webserver

The webserver provides an easy-to-use web interface to a part of the functionality of bio_embeddings. You can run the
 webserver with docker. You need
 
 * A mongodb container 
 * A webserver container
 * A worker container (celery)
 * An ampq broker (rabbitmq)
  
The worker should run on a host with a GPU.

You need to configure `CELERY_BROKER_URL`, `MONGO_URL` and `MODEL_DIRECTORY`. The `MODEL_DIRECTORY` has the following structure and content:

```
├── bert
│   └── model_directory
│       ├── bert_vocab_model.model
│       ├── config.json
│       └── pytorch_model.bin
├── bert_from_publication_annotations_extractors
│   ├── secondary_structure_checkpoint_file
│   └── subcellular_location_checkpoint_file
├── seqvec
│   ├── options_file
│   └── weights_file
└── secvec_from_publication_annotations_extractors
    ├── secondary_structure_checkpoint_file
    └── subcellular_location_checkpoint_file
```


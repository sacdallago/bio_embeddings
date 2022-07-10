# The bio_embeddings webserver

The webserver provides an easy-to-use web interface to a part of the functionality of bio_embeddings. We recommend running it through docker. For a full setup, you need
 
 * mongodb
 * rabbitmq
 * The webserver
 * Some workers, which should ideally run on a GPU host!

You need to provide the worker with a model directory, either by mounting it to `/mnt/models` (docker only) or by setting `MODEL_DIRECTORY` to the location. You can download it from https://rostlab.org/~bio_embeddings/webserver_models.zip. After unzipping, it should look like this:

```
├── bindembed21
│   ├── checkpoint1.pt
│   ├── checkpoint2.pt
│   ├── checkpoint3.pt
│   ├── checkpoint4.pt
│   └── checkpoint5.pt
├── colabfold
│   ├── LICENSE
│   └── params
│       ├── params_model_1.npz
│       ├── params_model_1_ptm.npz
│       ├── params_model_2.npz
│       ├── params_model_2_ptm.npz
│       ├── params_model_3.npz
│       ├── params_model_3_ptm.npz
│       ├── params_model_4.npz
│       ├── params_model_4_ptm.npz
│       ├── params_model_5.npz
│       └── params_model_5_ptm.npz
├── goa
│   ├── goa_annotations_2022_bpo.txt
│   ├── goa_annotations_2022_cco.txt
│   ├── goa_annotations_2022_mfo.txt
│   └── prott5_reference_embeddings.h5
├── light_attention
│   ├── la_prott5_solubility
│   └── la_prott5_subcellular_location
├── prottrans_t5_xl_u50
│   └── half_model_directory
│       ├── config.json
│       ├── pytorch_model.bin
│       └── spiece.model
├── t5_xl_u50_from_publication_annotations_extractors
│   ├── secondary_structure_checkpoint_file
│   └── subcellular_location_checkpoint_file
└── tmbed
    ├── cv_0.pt
    ├── cv_1.pt
    ├── cv_2.pt
    ├── cv_3.pt
    └── cv_4.pt
```

If you run without docker-compose, you need to configure `CELERY_BROKER_URL`, `MONGO_URL` and `MODEL_DIRECTORY` for the worker and the webserver.

## Production setup

We provide a configuration template at [docker-compose.prod.yml](docker-compose.prod.yml), which you can adapt to your needs. 

## Development Quickstart

Install docker and docker-compose.

[Download the weights](https://rostlab.org/~bio_embeddings/webserver_models.zip) and unzip them to `${HOME}/.cache/bio_embeddings`. You can also choose another location, but then you need to replace `${HOME}/.cache/bio_embeddings` with your path in [docker-compose.dev.yml](docker-compose.dev.yml) twice. Make sure to checkout the [docker compose documentation](https://docs.docker.com/compose/) to learn about the various options in docker compose files.

To start everything, run in the project root:

```shell script
docker-compose -f webserver/docker-compose.dev.yml up
```
 
The webserver is now ready at [localhost:3000](http://localhost:3000). If you add `bio_embeddings.local` to your hosts file, you can use an https version with a self-signed certificate at [https://bio_embeddings.local](https://bio_embeddings.local).

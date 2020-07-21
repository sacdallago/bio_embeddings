# Bio Embeddings
The project includes:

- A pipeline that allows to embed a FASTA file choosing from various embedders (see below), and then project and visualize the embeddings on 3D plots.
- General purpose library to embed protein sequences in any python app.
- A web server that takes in sequences, embeds them and returns the embeddings OR visualizes the embedding spaces on interactive plots online.

We presented the bio_embeddings pipeline as a talk at ISMB 2020. You can [find it on YouTube](https://www.youtube.com/watch?v=NucUA0QiOe0&feature=youtu.be), and a copy of the poster will soon be available on [F1000](https://f1000research.com/).

## News (current development cycle)

- Develop now includes new models from [ProtTrans](https://doi.org/10.1101/2020.07.12.199554). The models are `albert`, `bert` and `xlnet`. They will officially be included in release `0.1.4`, but can be installed by installing the pipeline from GitHub (see _Install Guides_)

## Install guides

You can install the pipeline via pip like so:

```bash
pip install bio-embeddings
```

To get the latest features, please install the pipeline like so:

```bash
pip install -U git+https://github.com/sacdallago/bio_embeddings.git
```

For some language models, additional dependencies are needed. We make it easy for you to install them by running the following additional `pip install` commands **after having installed the pipeline**:
- XLnet
  ```
  pip install bio_embeddings[xlnet]
  ```

## What model is right for you?

Each models has its strengths and weaknesses (speed, specificity, memory footprint...). There isn't a "one-fits-all" and we encourage you to at least try two different models when attempting a new exploratory project.

The models `albert`, `bert`, `seqvec` and `xlnet` were all trained with the goal of systematic predictions. From this pool, we believe the optimal model to be `bert`, followed by `seqvec`, which has been established for longer and uses a different principle (LSTM vs Transformer).

## Examples

We highly recommend you to check out the `examples` folder for pipeline examples, and the `notebooks` folder for post-processing pipeline runs and general purpose use of the embedders.

After having installed the package, you can:

1. Use the pipeline like:

    ```bash
    bio_embeddings config.yml
    ```

    A blueprint of the configuration file, and an example setup can be found in the `examples` directory of this repository.

1. Use the general purpose embedder objects via python, e.g.:

    ```python
    from bio_embeddings import SeqVecEmbedder

    embedder = SeqVecEmbedder()

    embedding = embedder.embed("SEQVENCE")
    ```

    More examples can be found in the `notebooks` folder of this repository.

## Development status

<details>
<summary>Pipeline stages</summary>
<br>

- embed:
  - [x] Bert (https://doi.org/10.1101/2020.07.12.199554)
  - [x] SeqVec (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3220-8)
  - [x] Albert (https://doi.org/10.1101/2020.07.12.199554)
  - [x] XLNet (https://doi.org/10.1101/2020.07.12.199554)
  - [ ] Fastext
  - [ ] Glove
  - [ ] Word2Vec
  - [ ] UniRep (https://www.nature.com/articles/s41592-019-0598-1?sfns=mo)
- project:
  - [x] t-SNE
  - [x] UMAP
</details>

<details>
<summary>Web server (unpublished)</summary>
<br>

- [x] SeqVec
- [x] Albert (https://doi.org/10.1101/2020.07.12.199554)
</details>

<details>
<summary>General purpose embedders</summary>
<br>

- [x] SeqVec (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3220-8)
- [x] Fastext
- [x] Glove
- [x] Word2Vec
- [ ] UniRep
- [x] Albert (https://doi.org/10.1101/2020.07.12.199554)
- [x] Bert (https://doi.org/10.1101/2020.07.12.199554)
- [x] XLNet (https://doi.org/10.1101/2020.07.12.199554)
</details>

## Building a Distribution
Building the packages best happens using invoke.
If you manganage your dependecies with poetry this should be already installed.
Simply use `poetry run invoke clean build` to update your requirements according to your current status
and to generate the dist files

### Additional dependencies and steps to run the webserver

If you want to run the webserver locally, you need to have some python backend deployment experience.
You'll need a couple of dependencies if you want to run the webserver locally: `pip install dash celery pymongo flask-restx pyyaml`.

Additionally, you will need to have two instances of the app run (the backend and at least one celery worker), and both instances must be granted access to a MongoDB and a RabbitMQ or Redis store for celery.

## Contributors

- Christian Dallago (lead)
- Konstantin Schütze
- Tobias Olenyi
- Michael Heinzinger

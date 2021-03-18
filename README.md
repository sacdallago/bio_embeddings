<p align="center">
  <a href="https://chat.bioembeddings.com/">
    <img src="https://chat.bioembeddings.com/api/v1/shield.svg?type=online&name=chat&icon=false" />
  </a>
</p>

# Bio Embeddings
Resources to learn about bio_embeddings:

- Quickly predict protein structure and function from sequence via embeddings: [embed.protein.properties](https://embed.protein.properties).
- Read the current documentation: [docs.bioembeddings.com](https://docs.bioembeddings.com).
- Chat with us: [chat.bioembeddings.com](https://chat.bioembeddings.com).
- We presented the bio_embeddings pipeline as a talk at ISMB 2020 & LMRL 2020. You can [find the talk on YouTube](https://www.youtube.com/watch?v=NucUA0QiOe0&feature=youtu.be), and [the poster on F1000](https://f1000research.com/posters/9-876).
- Check out the [`examples`](examples) of pipeline configurations a and [`notebooks`](notebooks).

Project aims:

  - Facilitate the use of language model based biological sequence representations for transfer-learning by providing a single, consistent interface and close-to-zero-friction
  - Reproducible workflows
  - Depth of representation (different models from different labs trained on different dataset for different purposes)
  - Extensive examples, handle complexity for users (e.g. CUDA OOM abstraction) and well documented warnings and error messages.

The project includes:

- General purpose python embedders based on open models trained on biological sequence representations (SeqVec, ProtTrans, UniRep,...)
- A pipeline which:
  - embeds sequences into matrix-representations (per-amino-acid) or vector-representations (per-sequence) that can be used to train learning models or for analytical purposes
  - projects per-sequence embedidngs into lower dimensional representations using UMAP or t-SNE (for lightwieght data handling and visualizations)
  - visualizes low dimensional sets of per-sequence embeddings onto 2D and 3D interactive plots (with and without annotations)
  - extracts annotations from per-sequence and per-amino-acid embeddings using supervised (when available) and unsupervised approaches (e.g. by network analysis)
- A webserver that wraps the pipeline into a distributed API for scalable and consistent workfolws

## Installation

You can install `bio_embeddings` via pip or use it via docker.

### Pip

Install the pipeline like so:

```bash
pip install bio-embeddings[all]
```

To get the latest features, please install the pipeline like so:

```bash
pip install -U "bio-embeddings[all] @ git+https://github.com/sacdallago/bio_embeddings.git"
```

### Docker

We provide a docker image at `ghcr.io/bioembeddings/bio_embeddings`. Simple usage example:

```shell_script
docker run --rm --gpus all \
    -v "$(pwd)/examples/docker":/mnt \
    -v bio_embeddings_weights_cache:/root/.cache/bio_embeddings \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    ghcr.io/bioembeddings/bio_embeddings:v0.1.6 /mnt/config.yml
```

See the [`docker`](examples/docker) example in the [`examples`](examples) folder for instructions. You can also use `ghcr.io/bioembeddings/bio_embeddings:latest` which is built from the latest commit.

### Installation notes:

`bio_embeddings` was developed for unix machines with GPU capabilities and [CUDA](https://developer.nvidia.com/cuda-zone) installed. If your setup diverges from this, you may encounter some inconsitencies (e.g. speed is significantly affected by the absence of a GPU and CUDA). For Windows users, we strongly recommend the use of [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10).


## What model is right for you?

Each models has its strengths and weaknesses (speed, specificity, memory footprint...). There isn't a "one-fits-all" and we encourage you to at least try two different models when attempting a new exploratory project.

The models `prottrans_bert_bfd`, `prottrans_albert_bfd`, `seqvec` and `prottrans_xlnet_uniref100` were all trained with the goal of systematic predictions. From this pool, we believe the optimal model to be `prottrans_bert_bfd`, followed by `seqvec`, which has been established for longer and uses a different principle (LSTM vs Transformer).

## Usage and examples

We highly recommend you to check out the [`examples`](examples) folder for pipeline examples, and the [`notebooks`](notebooks) folder for post-processing pipeline runs and general purpose use of the embedders.

After having installed the package, you can:

1. Use the pipeline like:

    ```bash
    bio_embeddings config.yml
    ```

    [A blueprint of the configuration file](examples/parameters_blueprint.yml), and an example setup can be found in the [`examples`](examples) directory of this repository.

1. Use the general purpose embedder objects via python, e.g.:

    ```python
    from bio_embeddings.embed import SeqVecEmbedder

    embedder = SeqVecEmbedder()

    embedding = embedder.embed("SEQVENCE")
    ```

    More examples can be found in the [`notebooks`](notebooks) folder of this repository.
    
## Cite

While we are working on a proper publication, if you are already using this tool, we would appreciate if you could cite the following poster:

> Dallago C, Schütze K, Heinzinger M et al. bio_embeddings: python pipeline for fast visualization of protein features extracted by language models [version 1; not peer reviewed]. F1000Research 2020, 9(ISCB Comm J):876 (poster) (doi: [10.7490/f1000research.1118163.1](https://doi.org/10.7490/f1000research.1118163.1))

## Contributors

- Christian Dallago (lead)
- Konstantin Schütze
- Tobias Olenyi
- Michael Heinzinger

----

## Development status


<details>
<summary>Pipeline stages</summary>
<br>

- embed:
  - [x] ProtTrans BERT trained on BFD (https://doi.org/10.1101/2020.07.12.199554)
  - [x] SeqVec (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3220-8)
  - [x] ProtTrans ALBERT trained on BFD (https://doi.org/10.1101/2020.07.12.199554)
  - [x] ProtTrans XLNet trained on UniRef100 (https://doi.org/10.1101/2020.07.12.199554)
  - [ ] Fastext
  - [ ] Glove
  - [ ] Word2Vec
  - [x] UniRep (https://www.nature.com/articles/s41592-019-0598-1)
  - [x] ESM/ESM1b (https://www.biorxiv.org/content/10.1101/622803v3)
  - [x] PLUS (https://github.com/mswzeus/PLUS/)
  - [x] CPCProt (https://www.biorxiv.org/content/10.1101/2020.09.04.283929v1.full.pdf)
- project:
  - [x] t-SNE
  - [x] UMAP
- visualize:
  - [x] 2D/3D sequence embedding space
- extract:
  - supervised:
    - [x] SeqVec: DSSP3, DSSP8, disorder, subcellular location and membrane boundness as in https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3220-8
    - [x] Bert: DSSP3, DSSP8, disorder, subcellular location and membrane boundness as in https://doi.org/10.1101/2020.07.12.199554
  - unsupervised:
    - [x] via sequence-level (reduced_embeddings), pairwise distance (euclidean like [goPredSim](https://github.com/Rostlab/goPredSim), more options available, e.g. cosine)
</details>

<details>
<summary>Web server (unpublished)</summary>
<br>

- [x] SeqVec supervised predictions
- [x] Bert supervised predictions
- [ ] SeqVec unsupervised predictions for GO: CC, BP,..
- [ ] Bert unsupervised predictions for GO: CC, BP,..
- [ ] SeqVec unsupervised predictions for SwissProt (just a link to the 1st-k-nn)
- [ ] Bert unsupervised predictions for SwissProt (just a link to the 1st-k-nn)
</details>

<details>
<summary>General purpose embedders</summary>
<br>

- [x] ProtTrans BERT trained on BFD (https://doi.org/10.1101/2020.07.12.199554)
- [x] SeqVec (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3220-8)
- [x] ProtTrans ALBERT trained on BFD (https://doi.org/10.1101/2020.07.12.199554)
- [x] ProtTrans XLNet trained on UniRef100 (https://doi.org/10.1101/2020.07.12.199554)
- [x] Fastext
- [x] Glove
- [x] Word2Vec
- [x] UniRep (https://www.nature.com/articles/s41592-019-0598-1)
- [x] ESM/ESM1b (https://www.biorxiv.org/content/10.1101/622803v3)
- [x] PLUS (https://github.com/mswzeus/PLUS/)
- [x] CPCProt (https://www.biorxiv.org/content/10.1101/2020.09.04.283929v1.full.pdf)
</details>

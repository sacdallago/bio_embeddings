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
- We presented the bio_embeddings pipeline as a talk at ISMB 2020 & LMRL 2020. You can [find the talk on YouTube](https://www.youtube.com/watch?v=NucUA0QiOe0&feature=youtu.be), [the poster on F1000](https://f1000research.com/posters/9-876), and our [Current Protocol Manuscript](https://doi.org/10.1002/cpz1.113).
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

You can install `bio_embeddings` via pip or use it via docker. Mind the additional dependencies for `align`.

### Pip

Install the pipeline **and all extras** like so:

```bash
pip install bio-embeddings[all]
```

To install the unstable version, please install the pipeline like so:

```bash
pip install -U "bio-embeddings[all] @ git+https://github.com/sacdallago/bio_embeddings.git"
```

If you only need to run a specific model (e.g. an ESM or ProtTrans model) you can install bio-embeddings without dependencies and then install the model-specific dependency, e.g.:
```bash
pip install bio-embeddings
pip install bio-embeddings[prottrans]
```

The extras are:
- seqvec
- prottrans
  - prottrans_albert_bfd
  - prottrans_bert_bfd
  - prottrans_t5_bfd
  - prottrans_t5_uniref50
  - prottrans_t5_xl_u50
  - prottrans_xlnet_uniref100
- esm
  - esm
  - esm1b
  - esm1v
- unirep
- cpcprot
- plus
- bepler
- deepblast

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

### Dependencies

To use the `mmseqs_search` protocol, or the `mmsesq2` functions in `align`, you additionally need to have [mmseqs2](https://mmseqs.com) in your path.

### Installation notes

`bio_embeddings` was developed for unix machines with GPU capabilities and [CUDA](https://developer.nvidia.com/cuda-zone) installed. If your setup diverges from this, you may encounter some inconsistencies (e.g. speed is significantly affected by the absence of a GPU and CUDA). For Windows users, we strongly recommend the use of [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10).


## What model is right for you?

Each models has its strengths and weaknesses (speed, specificity, memory footprint...). There isn't a "one-fits-all" and we encourage you to at least try two different models when attempting a new exploratory project.

The models `prottrans_t5_xl_u50`, `esm1b`, `esm`, `prottrans_bert_bfd`, `prottrans_albert_bfd`, `seqvec` and `prottrans_xlnet_uniref100` were all trained with the goal of systematic predictions. From this pool, we believe the optimal model to be `prottrans_t5_xl_u50`, followed by `esm1b`.

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

If you use `bio_embeddings` for your research, we would appreciate it if you could cite the following paper:

> Dallago, C., Schütze, K., Heinzinger, M., Olenyi, T., Littmann, M., Lu, A. X., Yang, K. K., Min, S., Yoon, S., Morton, J. T., & Rost, B. (2021). Learned embeddings from deep learning to visualize and predict protein sets. Current Protocols, 1, e113. doi: [10.1002/cpz1.113](https://doi.org/10.1002/cpz1.113)


The corresponding bibtex:
```
@article{https://doi.org/10.1002/cpz1.113,
author = {Dallago, Christian and Schütze, Konstantin and Heinzinger, Michael and Olenyi, Tobias and Littmann, Maria and Lu, Amy X. and Yang, Kevin K. and Min, Seonwoo and Yoon, Sungroh and Morton, James T. and Rost, Burkhard},
title = {Learned Embeddings from Deep Learning to Visualize and Predict Protein Sets},
journal = {Current Protocols},
volume = {1},
number = {5},
pages = {e113},
keywords = {deep learning embeddings, machine learning, protein annotation pipeline, protein representations, protein visualization},
doi = {https://doi.org/10.1002/cpz1.113},
url = {https://currentprotocols.onlinelibrary.wiley.com/doi/abs/10.1002/cpz1.113},
eprint = {https://currentprotocols.onlinelibrary.wiley.com/doi/pdf/10.1002/cpz1.113},
year = {2021}
}

Additionally, we invite you to cite the work from others that was collected in `bio_embeddings` (see section _"Tools by category"_ below). We are working on an enhanced user guide which will include proper references to all citable work collected in `bio_embeddings`.

```

## Contributors

- Christian Dallago (lead)
- Konstantin Schütze
- Tobias Olenyi
- Michael Heinzinger

Want to add your own model? See [contributing](https://docs.bioembeddings.com/latest/contributing/index.html) for instructions.

## Non-exhaustive list of tools available (see following section for more details):

- Fastext
- Glove
- Word2Vec
- SeqVec (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3220-8)
  - SeqVecSec and SeqVecLoc for secondary structure and subcellularlocalization prediction
- ProtTrans (ProtBert, ProtAlbert, ProtT5) (https://doi.org/10.1101/2020.07.12.199554)
  - ProtBertSec and ProtBertLoc for secondary structure and subcellular localization prediction
- UniRep (https://www.nature.com/articles/s41592-019-0598-1)
- ESM/ESM1b (https://www.biorxiv.org/content/10.1101/622803v3)
- PLUS (https://github.com/mswzeus/PLUS/)
- CPCProt (https://www.biorxiv.org/content/10.1101/2020.09.04.283929v1.full.pdf)
- PB-Tucker (https://www.biorxiv.org/content/10.1101/2021.01.21.427551v1)
- GoPredSim (https://www.nature.com/articles/s41598-020-80786-0)
- DeepBlast (https://www.biorxiv.org/content/10.1101/2020.11.03.365932v1)

## Datasets 

- `prottrans_t5_xl_u50` residue and sequence embeddings of the **Human proteome** at full precision + secondary structure predictions + sub-cellular localisation predictions: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5047020.svg)](https://doi.org/10.5281/zenodo.5047020)
- `prottrans_t5_xl_u50` residue and sequence embeddings of the **Fly proteome** at full precision + secondary structure predictions + sub-cellular localisation predictions + conservation prediction + variation prediction: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6322184.svg)](https://doi.org/10.5281/zenodo.6322184)

----

## Tools by category


<details>
<summary>Pipeline</summary>
<br>

- align:
  - DeepBlast (https://www.biorxiv.org/content/10.1101/2020.11.03.365932v1)
- embed:
  - ProtTrans BERT trained on BFD (https://doi.org/10.1101/2020.07.12.199554)
  - SeqVec (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3220-8)
  - ProtTrans ALBERT trained on BFD (https://doi.org/10.1101/2020.07.12.199554)
  - ProtTrans XLNet trained on UniRef100 (https://doi.org/10.1101/2020.07.12.199554)
  - ProtTrans T5 trained on BFD (https://doi.org/10.1101/2020.07.12.199554)
  - ProtTrans T5 trained on BFD and fine-tuned on UniRef50 (in-house)
  - UniRep (https://www.nature.com/articles/s41592-019-0598-1)
  - ESM/ESM1b (https://www.biorxiv.org/content/10.1101/622803v3)
  - PLUS (https://github.com/mswzeus/PLUS/)
  - CPCProt (https://www.biorxiv.org/content/10.1101/2020.09.04.283929v1.full.pdf)
- project:
  - t-SNE
  - UMAP
  - PB-Tucker (https://www.biorxiv.org/content/10.1101/2021.01.21.427551v1)
- visualize:
  - 2D/3D sequence embedding space
- extract:
  - supervised:
    - SeqVec: DSSP3, DSSP8, disorder, subcellular location and membrane boundness as in https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3220-8
    - ProtBertSec and ProtBertLoc as reported in https://doi.org/10.1101/2020.07.12.199554
  - unsupervised:
    - via sequence-level (reduced_embeddings), pairwise distance (euclidean like [goPredSim](https://github.com/Rostlab/goPredSim), more options available, e.g. cosine)
</details>

<details>
<summary>General purpose embedders</summary>
<br>

- ProtTrans BERT trained on BFD (https://doi.org/10.1101/2020.07.12.199554)
- SeqVec (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3220-8)
- ProtTrans ALBERT trained on BFD (https://doi.org/10.1101/2020.07.12.199554)
- ProtTrans XLNet trained on UniRef100 (https://doi.org/10.1101/2020.07.12.199554)
- ProtTrans T5 trained on BFD (https://doi.org/10.1101/2020.07.12.199554)
- ProtTrans T5 trained on BFD + fine-tuned on UniRef50 (https://doi.org/10.1101/2020.07.12.199554)
- Fastext
- Glove
- Word2Vec
- UniRep (https://www.nature.com/articles/s41592-019-0598-1)
- ESM/ESM1b (https://www.biorxiv.org/content/10.1101/622803v3)
- PLUS (https://github.com/mswzeus/PLUS/)
- CPCProt (https://www.biorxiv.org/content/10.1101/2020.09.04.283929v1.full.pdf)
</details>

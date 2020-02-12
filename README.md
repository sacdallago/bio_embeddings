# Bio Embeddings
The project includes:

- A pipeline that allows to embed a FASTA file with different embedders, and then call feature extractors (eg. get secondary structure) on top of the embeddings.
- A web server that takes in sequences, embeds them and returns the embeddings OR the feature vectors.
- General purpose library to embed protein sequences in any python app.

### Install guides

For now, this project is in beta. You can install the package via PIP like so:

```bash
pip install git+https://github.com/sacdallago/bio_embeddings.git@pipeline
```

### Examples

After having installed the package, you can

1. Use the pipeline like:

    ```bash
    bio_embeddings config.yml
    ```

    A blueprint of the configuration file, and an example setup can be found in the `examples` directory of this repository.

1. Use the general purpose embedder objects via python, e.g.:

    ```python
    from bio_embeddings import SeqVecEmbedder

    embedder = SeqVecEmbedder

    embedding = embedder.embed("SEQVENCE")
    ```

    More examples can be found in the `notebooks` folder of this repository.
### Development

1. Pipeline types
    - Embedders:   
        - [x] SeqVec v1/v2
        - [ ] TransformerXL
        - [ ] Fastext
        - [ ] Glove
        - [ ] Word2Vec
        - [ ] UniRep
    - Feature extractors:
        - SeqVec v1
            - [ ] DSSP8
            - [ ] DSSP3
            - [ ] Disorder
            - [ ] Subcell loc
            - [ ] Membrane boundness
1. Web server:  
    - [ ] SecVec
    - [ ] UniRep
    
1. General purpose objects:
    - [x] SecVec
    - [x] TransformerXL
    - [x] Fastext
    - [x] Glove
    - [x] Word2Vec
    - [x] UniRep
  
### Next:

- Add embedders in pipeline
- Make general webserver + webservice

### Improvements needed:

- Accept only natural sequences?

### Wanna use it now?
  
Use the `notebooks` folder, that will always include the latest version of the src. Note: although this is in alpha, we will try to keep the API consistent.

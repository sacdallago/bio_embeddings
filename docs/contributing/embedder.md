# Add a new language model/embedder

* Pick a name, which should be the one you're using in the publication, and a lowercase version with underscores (snake_case). E.g. for one hot encoding, we use `one_hot_encoding`. The class name is the CamelCase version, in this case `OneHotEncodingEmbedder`. Stay consistent where you place the underscores.
* Add all new dependencies in `pyproject.toml` in a new extra
* Add an entry to `bio_embeddings/utilities/defaults.yml` with a link to the weights.
* Create a new class in `bio_embeddings/embed` that at least implements `EmbedderInterface`, or even better (for GPU based models) `EmbedderWithFallback`. The most simple example is `OneHotEncodingEmbedder`, are more realistic example is `ProtTransT5Embedder` and its subclasses. If you add any new options, add them to `KNOWN_EMBED_OPTIONS` in `bio_embeddings/embed/pipeline.py`
* Add the class in `bio_embeddings/embed/__init__.py`
* The following two are checked by `SKIP_SLOW_TESTS=1 pytest`:
    * Add the model size in the docs of `bio_embeddings/embed/__init__.py`
    * Add it to `DEFAULT_MAX_AMINO_ACIDS`
* Add it to the tests following the instructions in `tests/test_embedder_embedding.py`
* Write a pipeline with your embedder, see that it works
* Send a pull request 🚀
 
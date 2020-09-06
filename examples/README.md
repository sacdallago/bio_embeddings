# Scope

In this folder you will find a couple examples of how to use the pipeline or its outputs.

You will also find the `parameters_blueprint.yml` file. This file contains all possible parameters for the pipeline with details about their functionality.


### Ready to run pipeline examples:

For each of the following examples, `cd` in the directory (e.g. `cd use_case_one`) and execute `bio_embeddings config.yml`.

- A simple way to visualize embeddings, `use_case_one`

  Use case: you have a set of proteins (in FASTA format) and want to create amino acid-level embeddings, as well as protein-level embeddings.
  Additionally, you have an annotation file with some property for a subset of the proteins in your dataset. For these, you want to produce a visualization of the sequences and how they separate in space.
  In this example, the sequence-level embeddings (`reduced_embeddings_file`) are leveraged to compute t-SNE projections, which are then used to generate a 3D plot of the "sequence space".
  
  Noteworthy files produced:
    - The `embed` stage produces an `embeddings_file` and a `reduced_embeddings_file`.
    The former contains embeddings for each AA in each sequence, while the latter contains (fixed sized) embeddings for each sequence in your set.
    You can use the notebooks to check out how to open these. 
    - The `project` stage produces a CSV `projected_embeddings_file`, which contains `(x,y,z)` coordinates for each sequence in your set.
    - The `visualize` stage produces an HTML `plot_file` containing the plot of the sequences derived from the projection's coordinates.

- Same as before, but using cached weights, which is faster: `use_case_two`

  Use case: you have a set of proteins (in FASTA format) and want to create amino acid-level embeddings, as well as protein-level embeddings.
  Additionally, you have an annotation file with some property for a subset of the proteins in your dataset. For these, you want to produce a visualization of the sequences and how they separate in space.
  This time around: you downloaded the models locally (faster execution) and want to provide the path to the model's weights and options.
  You also annotated your proteins using an md5 hash of the sequence instead of arbitrary identifiers.

- How can I display 3D sequence spaces from embeddings? `use_case_three`

  Use case: you already have per-protein embeddings of a certain dataset and want to produce various t-sne plots, using both different annotation files and different t-sne parameters.

  Files you will need:

    - Reduced embeddings file: a per-protein embedding file in hdf5 format
    - Mapping file: a file containing a mapping from md5 hash of the sequence to an arbitrary identifier (e.g. the one used in an annotation file)
    - Annotation file(s): CSV files containing annotations for the proteins in the reduced embeddings file

  *Note*: While it is possible to use the pipeline to produce many visualizations for many different annotations, it may be more efficient to use a Notebook for this.
  We include a notebook (`project_visualize_pipeline_embeddings`) covering the same use case as the one presented here in the `notebooks` folder at the root of this project.

- Trained supervised models: get protein structure and function annotations, `supervised_annotation_extraction`

  Use case: you have a set of proteins (in FASTA format) and want to extract features using the supervised models published during evaluation of SeqVec and Bert (aka: DSSP3, DSSP8, disorder, localization and membrane vs. soluble).
  
  Noteworthy files produced:
    - The `extract` stages produce
       - `DSSP3_predictions_file`, `DSSP8_predictions_file`, and `disorder_predictions_file`, which are FASTA files containing the respective, per-AA annotations;
       - additionally a CSV `per_sequence_predictions_file` contains per-sequence annotations, aka: localization and if a sequence is predicted to be membrane-bound or not.
    

- Transfer annotations from labeled sequences to unlabeled sequences: `unsupervised_annotation_extraction`

  Use case: you have a set of proteins with known properties (we call this "`reference`"), and you have a set of proteins for which you would like to infer these properties.
  Unsupervised annotation extraction (also annotation transfer) happens through k-nearest-neighbour search of the closest embeddings in a reference, annotated dataset.
  Distances between input sequences and reference dataset are calculated via pairwise distances between target (your input sequences) and reference embeddings (e.g. SwissProt).
  The pipeline's implementation is inspired by [goPredSim](https://github.com/Rostlab/goPredSim) and offers standard distance metrics, e.g. euclidean, manhattan, and also pseudo distances e.g. cosine
  In this example, we use the `reduced_embeddings_file` calculated in `disprot`, and annotations from the CSV file there to transfer annotations onto an unknown dataset.
  
   Noteworthy files produced:
     - The `extract` stages produces:
         - a CSV `pairwise_distances_matrix_file`, which contains all pairwise distances (euclidean and cosine in this example) between input sequences/embeddings and reference embeddings;
         - a CSV `transferred_annotations_file`, which contains a column with the transferred annotations, and k columns with the k-th closest element its distance, identifier and annotations.
  

- `cath`, used for our manuscript

  This example includes sequences pulled directly from the CATH database (http://www.cathdb.info) and annotations for structural folds. To reduce the embeddings, UMAP was used.
  Note: since the FASTA file pulled from CATH contains duplicate sequences, the remapping has been set to "simple". This is *discouraged*, as it may lead to higher computation times (embedding the same sequence multiple times), and could lead to conflicts when overlaying annotations.

- `disprot`, used for our manuscript

  Similar to the `cath` example, but using the DisProt (https://www.disprot.org) database instead. Annotations contain "highly disorder" for proteins with >80% disorder, and "low disorder" for proteins with <20% disordered AA content.
  Note that in this example we exclude proteins with unknown annotation (see visualize stage in config).

- `docker`

  This example is similar to use_case_one, except that the paths are changed to work with docker (i.e. `/mnt` prefixes everywhere). From the project root, you can run it with:

  ```shell_script
  docker run --rm --gpus all \
      -v "$(pwd)/examples/docker":/mnt \
      -u $(id -u ${USER}):$(id -g ${USER}) \
      rostlab/bio_embeddings /mnt/config.yml
  ```

  In general, you should mount all input files into `/mnt`, e.g. you might need to add something like `-v /nfs/my_sequence_storage/proteomes.fasta:/mnt/sequences.fasta`. The `--gpus all` lets docker use the GPU and `-u $(id -u ${USER}):$(id -g ${USER})` makes sure that the results are owned by the current user and not by root.

  You'll find the results in `examples/docker/output`.
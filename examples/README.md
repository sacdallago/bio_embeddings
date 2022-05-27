# Pipeline Examples

In this folder you will find a couple examples of how to use the pipeline or its outputs.

You can refer to the [`parameters_blueprint.yml`](parameters_blueprint.yml) file for all pipeline parameters, including details about their functionality.


**Ready to run pipeline examples:**

For each of the following examples, `cd` in the directory (e.g. `cd use_case_one`) and execute `bio_embeddings config.yml`. All example outputs will also soon will be available for comparison purposes.

**In brief:**

|Folder|Purpose|
|---|---|
|[use_case_one](#a-simple-way-to-visualize-embeddings-use_case_one)|Embedding generation & visualization|
|[use_case_two](#same-as-before-but-using-cached-weights-which-is-faster-use_case_two)|Embedding generation & visualization|
|[use_case_three](#how-can-i-display-3d-sequence-spaces-from-embeddings-use_case_three)|Embedding visualization|
|[Supervised annotation extraction](#trained-supervised-models-get-protein-structure-and-function-annotations-supervised_annotation_extraction)|Prediction of localization and secondary structure|
|[Unsupervised annotation extraction](#transfer-annotations-from-labeled-sequences-to-unlabeled-sequences-unsupervised_annotation_extraction)|Transfer of annotations|
|[goPredSim](#transfer-go-annotations-gopredsim)|Transfer of annotations|
|[goPredSim using ProtTrans BERT BFD](#transfer-go-annotations-gopredsim_protbert)|Transfer of annotations|
|[deeploc](#deeploc-used-for-our-manuscript)|Embedding generation & visualization|
|[cath](#cath-used-for-our-manuscript)|Embedding generation & visualization|
|[disprot](#disprot-used-for-our-manuscript)|Embedding generation & visualization|
|[docker](#docker)|Pipeline use through Docker|
|[advanced_use_case](#pass-your-own-reducer-advanced_use_case)|Embedding generation & transformation|
|[deepblast](#deepblast)|Use [DeepBLAST](https://github.com/flatironinstitute/deepblast) to align sequences|
|[tucker](#tucker)|Shows how tucker embeddings better separates sequences by CATH class when compared to plain Bert embeddings|
|[light attention](#light-attention)|Embedding generation & Prediction of subcellular localization|
|[mutagenesis](#mutagenesis)|Prediction of mutation effects|
---

### A simple way to visualize embeddings, `use_case_one`

**Use case**: you have a set of proteins (in FASTA format) and want to create amino acid-level embeddings, as well as protein-level embeddings.
  Additionally, you have an annotation file with some property for a subset of the proteins in your dataset. For these, you want to produce a visualization of the sequences and how they separate in space.
  In this example, the sequence-level embeddings (`reduced_embeddings_file`) are leveraged to compute t-SNE projections, which are then used to generate a 3D plot of the "sequence space".
  
**Noteworthy files produced**:
  - The `embed` stage produces an `embeddings_file` and a `reduced_embeddings_file`.
  The former contains embeddings for each AA in each sequence, while the latter contains (fixed sized) embeddings for each sequence in your set.
  You can use the notebooks to check out how to open these. 
  - The `project` stage produces a h5 `projected_reduced_embeddings_file`, which contains `(x,y,z)` coordinates for each sequence in your set.
  - The `visualize` stage produces an HTML `plot_file` containing the plot of the sequences derived from the projection's coordinates.

---

### Same as before, but using cached weights, which is faster: `use_case_two`

**Use case**: you have a set of proteins (in FASTA format) and want to create amino acid-level embeddings, as well as protein-level embeddings.
  Additionally, you have an annotation file with some property for a subset of the proteins in your dataset. For these, you want to produce a visualization of the sequences and how they separate in space.
  This time around: you downloaded the models locally (faster execution) and want to provide the path to the model's weights and options.
  You also annotated your proteins using an md5 hash of the sequence instead of arbitrary identifiers.

---

### How can I display 3D sequence spaces from embeddings? `use_case_three`

**Use case**: you already have per-protein embeddings of a certain dataset and want to produce various t-sne plots, using both different annotation files and different t-sne parameters.

**Files you need**:
  - Reduced embeddings file: a per-protein embedding file in hdf5 format
  - Mapping file: a file containing a mapping from md5 hash of the sequence to an arbitrary identifier (e.g. the one used in an annotation file)
  - Annotation file(s): CSV files containing annotations for the proteins in the reduced embeddings file

**Note**: While it is possible to use the pipeline to produce many visualizations for many different annotations, it may be more efficient to use a Notebook for this.
We include a notebook (`project_visualize_pipeline_embeddings`) covering the same use case as the one presented here in the `notebooks` folder at the root of this project.

---

### Trained supervised models: get protein structure and function annotations, `supervised_annotation_extraction`

**Use case**: you have a set of proteins (in FASTA format) and want to extract features using the supervised models published during evaluation of ProtT5 (aka: DSSP3, DSSP8, disorder, localization and membrane vs. soluble).
  
**Noteworthy files produced**:
  - The `extract` stages produce
    - `DSSP3_predictions_file`, `DSSP8_predictions_file`, and `disorder_predictions_file`, which are FASTA files containing the respective, per-AA annotations;
    - additionally a CSV `per_sequence_predictions_file` contains per-sequence annotations, aka: localization and if a sequence is predicted to be membrane-bound or not.

---

### Transfer annotations from labeled sequences to unlabeled sequences: `unsupervised_annotation_extraction`

**Use case**: you have a set of proteins with known properties (we call this "`reference`"), and you have a set of proteins for which you would like to infer these properties.
Unsupervised annotation extraction (also annotation transfer) happens through k-nearest-neighbour search of the closest embeddings in a reference, annotated dataset.
Distances between input sequences and reference dataset are calculated via pairwise distances between target (your input sequences) and reference embeddings (e.g. SwissProt).
The pipeline's implementation is inspired by [goPredSim](https://github.com/Rostlab/goPredSim) and offers standard distance metrics, e.g. euclidean, manhattan, and also pseudo distances e.g. cosine
In this example, we use the `reduced_embeddings_file` calculated in `disprot`, and annotations from the CSV file there to transfer annotations onto an unknown dataset.
  
**Noteworthy files produced**:
  - The `extract` stages produces:
    - a CSV `pairwise_distances_matrix_file`, which contains all pairwise distances (euclidean and cosine in this example) between input sequences/embeddings and reference embeddings;
    - a CSV `transferred_annotations_file`, which contains a column with the transferred annotations, and k columns with the k-th closest element its distance, identifier and annotations.
   
   
---

### Transfer GO annotations: `goPredSim`

**Use case**: You have a set of proteins for which you would like to infer GO annotations for (as is done in [goPredSim](https://github.com/Rostlab/goPredSim)).
This uses the [unsupervised_annotation_extraction](#transfer-annotations-from-labeled-sequences-to-unlabeled-sequences-unsupervised_annotation_extraction) idea.

**Prerequisites**: 

 - for this example, you have to download the `seqvec_reference_embeddings.h5` and `annotations.csv` files from http://data.bioembeddings.com/public/embeddings/reference/goa/ , and store them in the same folder as the `config.yml`.
 - Our tests show that this experiment consumed 16.3GB of system RAM at peak computation (average: 2.5GB). The RAM requirements will increase with the number of sequences in your set.
  
**Noteworthy files produced**:
  - The `extract` stages produces:
    - a CSV `transferred_annotations_file`, which contains a column with the transferred GO terms.
    
---

### Transfer GO annotations: `goPredSim_protbert`

The same as [goPredSim](#transfer-go-annotations-gopredsim), but using `prottrans-bert-bfd` instead. You can find the reference protbert embeddings at: http://data.bioembeddings.com/public/embeddings/reference/goa/

---

### `deeploc`, used for our manuscript

This example includes sequences pulled directly from DeepLoc 1.0 (http://www.cbs.dtu.dk/services/DeepLoc/data.php). Annotations (`deeploc_annotations.csv`) were extracted from the FASTA headers.

**Note**: since the FASTA file pulled from DeepLoc contains duplicate sequences, the remapping has been set to "simple". This is generally *discouraged*, but for the sake of simplicity of this example has been kept "as-is".


---

### `cath`, used for our manuscript

This example includes sequences pulled directly from the CATH database (http://www.cathdb.info) and annotations for structural folds. To reduce the embeddings, UMAP was used.

**Note**: since the FASTA file pulled from CATH contains duplicate sequences, the remapping has been set to "simple". This is *discouraged*, as it may lead to higher computation times (embedding the same sequence multiple times), and could lead to conflicts when overlaying annotations.

---

### `disprot`, used for our manuscript

Similar to the `cath` example, but using the DisProt (https://www.disprot.org) database instead. Annotations contain "highly disorder" for proteins with >80% disorder, and "low disorder" for proteins with <20% disordered AA content.

**Note**: in this example we exclude proteins with unknown annotation (see visualize stage in config).

---

### `docker`

This example is similar to use_case_one, except that the paths are changed to work with docker (i.e. `/mnt` prefixes everywhere). From the project root, you can run it with:

```shell_script
docker run --rm --gpus all \
    -v "$(pwd)/examples/docker":/mnt \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    rostlab/bio_embeddings /mnt/config.yml
```

In general, you should mount all input files into `/mnt`, e.g. you might need to add something like `-v /nfs/my_sequence_storage/proteomes.fasta:/mnt/sequences.fasta`. The `--gpus all` lets docker use the GPU and `-u $(id -u ${USER}):$(id -g ${USER})` makes sure that the results are owned by the current user and not by root.
You'll find the results in `examples/docker/output`.

---

### `tucker`

We compare tucker to plain Bert for separating CATH classes. As baseline, we embed a small test set of CATH domains (which were excluded from tucker training) with Bert (`prottrans_bert_bfd`, to be exact) and plot those with umap and plotly. We then project the bert embeddings with tucker (`pb_tucker`) and plot them in the same way. 

**Noteworthy files produced**:
  - The `visualize` stages produce:
    - `tucker_cath/visualize_bert_class/plot_file.html` for plain Bert 
    - `tucker_cath/visualize_tucker_class/plot_file.html` for Bert projected with tucker

By comparing the two you can observe that tucker separates the mainly alpha and the mainly beta classes much clearer than plain Bert.

---

### Pass your own reducer, `advanced_use_case`

In some cases, you are interested in doing something else then mean pooling the embeddings for per-sequence representations. The pipeline has an experimental feature which offers you to directly transform the per-amino acid embeddings into a different format (e.g. you can max pool, you can do other types of transformations).
A small example extracting the first LSTM layer and mean pooling that (in the case of SeqVec) and max pooling instead of mean pooling (in the case of ProtTrans-BERT-BFD) is available in the advanced use case folder.

---

### Light Attention

Using a light attention mechanism to aggregate residue embeddings for protein sequences we trained supervised models to predict subcellular localization.

**Noteworthy files produced**:
 - `la_prott5` creates `per_sequence_predictions_file.csv`

---

### Mutagenesis

In-silico mutagenesis using ProtTrans-Bert-BFD. This computes the likelihood that, according to Bert, a residue in a protein can be a certain amino acid. This can be used as an estimate for the effect of a mutation.

**Noteworthy files produced**:
 - The `protbert_bfd_mutagenesis` protocol writes `residue_probabilities_file.csv` with probabilities for all sequences
 - `plot_mutagenesis` creates a separate interactive plot for each sequence as html file

**Note**: Mutagenesis is much slower than embedding, so only use it with few sequences. Internally, we have to rerun the entire model for each residue we want to predict, so we do the computation for each residue instead of once per protein.

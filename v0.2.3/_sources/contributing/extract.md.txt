# Implement a new predictor

If you have a predictor that predicts a certain feature from some embeddings, you can add it to the `extract` module of `bio_embeddings`. In the following is a short guide where “project_name.” refers to the abbreviation of your predictor, e.g. “ProtT5cons”, which refers to a CNN that predicts conservation from ProtT5-XL-U50 embeddings.

## Step 1: Class-to-label-mapping

You create a new file that defines the labels (e.g. {“H”,”E”,”L”} for helix, sheet, and other) that are mapped from your raw network output (e.g. three secondary structures classes in [0,2]).

This is done by creating a new “annotations” file that defines the mapping here: `bio_embeddings/extract/annotations/`.

Use an expressive name that describes the feature you are mapping, for example `conservation.py`.

Use existing files in this folder as template (this will hold true for all further steps). Watch out for templates that define tasks on at per-residue level (e.g. secondary structure or conservation) versus tasks on the per-protein level, i.e., modalities/features that define an entire protein (e.g. subcellular localization), and choose your template according to your task.

Additional things to look out for:

- When creating annotations at the per-residue, use single letters/numbers as value to the enum key:
  ```
  class Disorder(Enum):
    DISORDER = 'X'
    ORDER = '-'
    UNKNOWN = '?'
  ```
  this as opposed to longer values for per-protein features, e.g.:
  ```
  class Location(Enum):
    CELL_MEMBRANE = 'Cell-Membrane'
    CYTOPLASM = 'Cytoplasm'
    ENDOPLASMATIC_RETICULUM = 'Endoplasmic reticulum'
    GOLGI_APPARATUS = 'Golgi - Apparatus'
  ```
  the reason you want to do this is that when the predictions are exported in the pipeline, they are exported as a FASTA file. As such, if you have more than 1 character per residue, it will be impossible to map back and forth between the sequence and the predictions.

- Use the `def __str__(self):` method to define as complete as possible description of a particular predicted feature means, so that users wondering what a particular key value means can interpret it by simply printing it:
  ```
  [IN]  print(Conservation.cons_7)
  [OUT] Conserved. Score=7 (1: variable, 9: conserved)
  ```

### Step 1.1.: Bridging imports

Add the new annotation class that you’ve defined in the step before to the `__init__.py` of `bio_embeddings/extract/annotations/` (see file for examples)

## Step 2: Embedding-to-FeatureExtractor/Predictor-bridge

In this step, you create a new directory that holds all files that are necessary to

1. embed a protein sequence using the correct embedding,
2. run your pre-trained predictor on top of embeddings to derive predictions, and
3. use the mapping you defined in step 1 to map from raw class output (e.g. integer in [0,2]) to actual class labels (e.g. string in {“H”,”E”,”L”}).

### Step 2.1: File creation

Within the new directory, create a new file for your NN-architecture; use a combination of the feature you are predicting (conservation) and the architecture (CNN) as file name; for example, `conservation_cnn.py` holds the definition of the CNN-architecture.

Add the definition of your architecture to this new file. Use again code from other projects as template.

### Step 2.2: The meat of prediction

In the same directory, create a new file that defines how Embedders/LMs and Feature-extractors (predictors) should be stitched together to produce predictions.

For this, create a new file similar to `basic/BasicAnnotationExtractor.py` but adjust it to your labels/task. Name the new file  `{project_name}_annotation_extractor.py` (for example, `prot_t5_cons_annotation_extractor.py`).

This file will hold a class (e.g. `ProtT5consAnnotationExtractor`; you can use this one as template for residue-level predictions) that in turn will hold the logic to transform embeddings into predictions. In a simple way, the class loads the weights of your predictor architecture, receives embeddings via a certain function which are fed to the predictor to produce raw class predictions (e.g. in [0,2]), finally maps those to labels defined in the `annotations` file from Step 1 and returns the predictions in some structured form.

Note: use `neccessary_files` to define any files that you need to load to make the predictions. These files will also be made public (See Step 4). The name of the files must be the same as in `defaults.yaml`. The files **must** end in `_file` (e.g. `weights_file`, `blosum_scores_file`).

### Step 2.2: Bridging imports

Create in the same directory an `__init__.py` file that imports the class that you wrote in Step 2.2 (for example, for conservation prediction:

```
from bio_embeddings.extract.prott5cons.prot_t5_cons_annotation_extractor import ProtT5consAnnotationExtractor
```

### Step 3: Make your predictor part of the pipeline

Expand the `pipeline.py` file in `bio_embeddings/extract` to automate the function-call of the code you’ve defined in Step 2. Watch out for differences in per-residue and per-protein predictions and use existing functions as templates. The output of your predictions should either be a FASTA file (per-residue predictions) or a CSV file (per-protein predictions).

### Step 4: Add model weights

Last but not least, you need to upload the weights of your predictor to an FTP (http://data.bioembeddings.com/public/embeddings/feature_models/). This you do by pinging Chris/Konstantin through support [A!T] bioembeddings.com (if external) or Mattermost. Once the weights are online, add "project_name” to the `defaults.yaml` in `bio_embeddings/utilities`. This will enable automatic download of the weights through functionality already in `bio_embeddings`.

### Step 5: Write tests

In order to ensure that everything runs as expected you should also add a some simple sanity check to `bio_embeddings/tests`. There you can define, for example, the expected accuracy for a few proteins (or a single protein for per-residue task as every residue is a sample). Important is to pick proteins where you know ground truth labels. After defining the expected accuracy (the accuracy you got for this protein using your own code, not the bio_embeddings implementation), you can write an automated test (again, check other tests as template) that runs the new predictor that you’ve defined in the steps above on the test-protein of your choice. Those predictions can be used to compute an accuracy (bio_embeddings_accuracy) which in turn can be compared the expected accuracy. Alternatively, you can also directly compare predictions from your own code and predictions from your bio_embeddings
implementation.


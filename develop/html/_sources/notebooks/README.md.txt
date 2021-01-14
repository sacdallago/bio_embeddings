# Notebooks

The notebooks in this folder can be executed locally on your machine or on Google Colab (a tool that allows you to run code online). If you run the Notebooks on your own machine, you might ignore the "Colab Initialization" code, but you will have to download the files required by the notebook. If you do run the Notebooks on Colab, you have to execute commands to install the pipeline and download neccessary files, which are the blocks of code followin the "Colab Initialization" header.



##  Preface

From experience within our lab and with collaborators we have created a set of Notebooks that try to address different aspects of what is generally needed. The Notebooks presented here are to be viewed as "building blocks" for your exploratory projects! We've tried to keep the notebooks short and to the point. Often, you will need to grab a thing from here and a thing from there.

## From the manuscript

| Purpose                                                      | Colab                                                        | Notebook                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ | :----------------------------------------------------- |
| Basic Protocol 2 and alternates: use deeploc embeddings produced by the pipeline to plot sequence spaces. This is virtually the same as [this pipeline example](../examples/deeploc), but here we can tune the UMAP parameters until we obtain a nice graphic to put in a presentation :) . | [Click here](https://colab.research.google.com/github/sacdallago/bio_embeddings/blob/develop/notebooks/deeploc_visualizations.ipynb) | [Click here](deeploc_visualizations.ipynb)    |
| Basic Protocol 3: train a simple machine learning classifier to predict subcellular localizations training on [DeepLoc embeddings](../examples/deeploc). | [Click here](https://colab.research.google.com/github/sacdallago/bio_embeddings/blob/develop/notebooks/deeploc_machine_learning.ipynb) | [Click here](deeploc_machine_learning.ipynb)                      |



## Exploring modules from the `bio_embeddings` package

| Purpose                                                      | Colab                                                        | Notebook                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ | :----------------------------------------------------- |
| Use the general purpose embedding objects to embed a sequence passed as string | [Click here](https://colab.research.google.com/github/sacdallago/bio_embeddings/blob/develop/notebooks/embed_custom_sequence.ipynb) | [Click here](embed_custom_sequence.ipynb)              |
| Use `Bio` to load a FASTA file and the general purpose embedding objects to embed sequences from the file | [Click here](https://colab.research.google.com/github/sacdallago/bio_embeddings/blob/develop/notebooks/embed_fasta_sequences.ipynb) | [Click here](embed_fasta_sequences.ipynb)              |
| Embed a sequence and extract annotations using supervised models from `bio_embeddings.extract`. You can achieve the same results using the pipeline like in [this example](../examples/supervised_annotation_extraction). | [Click here](https://colab.research.google.com/github/sacdallago/bio_embeddings/blob/develop/notebooks/extract_supervised_from_seqvec.ipynb) | [Click here](extract_supervised_from_seqvec.ipynb)     |
| Embed a sequence and transfer GO annotations using unsupervised techniques found in `bio_embeddings.extract` (vistually the same as [goPredSim](https://github.com/Rostlab/goPredSim/)). You can achieve the same results using the pipeline like in [this example](../examples/goPredSim_prottrans_bert_bfd). | [Click here](https://colab.research.google.com/github/sacdallago/bio_embeddings/blob/develop/notebooks/goPredSim.ipynb) | [Click here](goPredSim.ipynb)                          |

## Proper use cases from collaborations
| Purpose                                                      | Colab                                                        | Notebook                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ | :----------------------------------------------------- |
| Embed a few sequences and try out different ideas to see if the embeddings are able to cluster different sequences | [Click here](https://colab.research.google.com/github/sacdallago/bio_embeddings/blob/develop/notebooks/project_visualize_custom_embedings.ipynb) | [Click here](project_visualize_custom_embedings.ipynb) |


## Exploring pipeline output files
| Purpose                                                      | Colab                                                        | Notebook                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | :----------------------------------------------------------- |
| Open an embedding file, the principal output of a pipeline run | [Click here](https://colab.research.google.com/github/sacdallago/bio_embeddings/blob/develop/notebooks/open_embedding_file.ipynb) | [Click here](open_embedding_file.ipynb)                      |
| Use embeddings produced by the pipeline to plot sequence spaces. This is virtually similar to using the `project` and `visualize` steps of the pipeline, but in this case, you can change paramteres on the fly in the Notebook. A similar pipeline example [here](../examples/use_case_three). | [Click here](https://colab.research.google.com/github/sacdallago/bio_embeddings/blob/develop/notebooks/project_visualize_pipeline_embeddings.ipynb) | [Click here](project_visualize_pipeline_embeddings.ipynb)    |
| Use deeploc embeddings produced by the pipeline to plot sequence spaces. This is virtually the same as [this pipeline example](../examples/deeploc), but here we can tune the UMAP parameters until we obtain a nice graphic to put in a presentation :) . | [Click here](https://colab.research.google.com/github/sacdallago/bio_embeddings/blob/develop/notebooks/deeploc_visualizations.ipynb) | [Click here](deeploc_visualizations.ipynb)    |
| Analyze embedding sets by studying their similarity and transferring annotations | [Click here](https://colab.research.google.com/github/sacdallago/bio_embeddings/blob/develop/notebooks/pairwise_distances_and_nearest_neighbours.ipynb) | [Click here](pairwise_distances_and_nearest_neighbours.ipynb) |
| Naïvely plot embeddings to distinguish patterns in your embedded sequences | [Click here](https://colab.research.google.com/github/sacdallago/bio_embeddings/blob/develop/notebooks/per_amino_acid_embedding_visualization.ipynb) | [Click here](per_amino_acid_embedding_visualization.ipynb)   |

## Advanced use cases
| Purpose                                                      | Colab                                                        | Notebook                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | :----------------------------------------------------------- |
| Train a simple machine learning classifier to predict subcellular localizations training on [DeepLoc embeddings](../examples/deeploc). | [Click here](https://colab.research.google.com/github/sacdallago/bio_embeddings/blob/develop/notebooks/deeploc_machine_learning.ipynb) | [Click here](deeploc_machine_learning.ipynb)                      |

## Utilities
| Purpose                                                      | Colab                                                        | Notebook                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | :---------------------------------------------------------- |
| Re-index an `embeddings_file` using your original identifiers. Useful when creating reference embedding sets for the unsupervised `extract` stage | [Click here](https://colab.research.google.com/github/sacdallago/bio_embeddings/blob/develop/notebooks/reindex_h5_file.ipynb) | [Click here](reindex_h5_file.ipynb)                         |
| Remove identifiers from an annotation file, useful when the pipeline suggest you to do so! :) | [Click here](https://colab.research.google.com/github/sacdallago/bio_embeddings/blob/develop/notebooks/remove_identifiers_from_annotation_file.ipynb) | [Click here](remove_identifiers_from_annotation_file.ipynb) |









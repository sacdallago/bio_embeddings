# Changelog

## 0.2.3

 * Fix missing docker images
 * Added a contributing guide

## 0.2.2

 * Added the `esm1v` embedder from [Meier et al. 2021](https://www.biorxiv.org/content/10.1101/2021.07.09.450648v1), which is part of facebook's [esm](https://github.com/facebookresearch/esm). Note that this is an ensemble model, so you need to pass `ensemble_id` with a value from 1 to 5 to select which weights to use.
 * Added the `bindEmbed21DL` extract protocol which is an ensemble of 5 convolutional neural network that predicts of 3 different types of binding residues (metal, nucleic acids, small molecules).
 * Fix model download
 * Update jaxlib to fix pip installation

## v0.2.1

 * BETA: in-silico mutagenesis using ProtTransBertBFD. This computes the likelihood that, according to Bert, a residue in a protein can be a certain amino acid, which can be used as an estimate for the effect of a mutation. This adds two a new `mutagenesis` and a new protocol `plot_mutagenesis` in the `visualize` stages, of which the first one computes the probabilities and writes them to a csv file while the latter visualizes the results as interactive plotly figure.
 * Support `half_precision_model` for `prottrans_bert_bfd` and `prottrans_albert_bfd`
 * Fix a `n_components: 2` in the plotly protocol

## v0.2.0

 * Added the `prottrans_t5_xl_u50`/`ProtTransT5XLU50Embedder` embedder from the latest ProtTrans revision. You should use this over `prottrans_t5_bfd` and `prottrans_t5_uniref50`. 
 * The `projected_embeddings_file.csv` of project stages has been renamed to `projected_reduced_embeddings_file.h5`. For backwards compatibility, `projected_embeddings_file.csv` is still written.
 * The `projected_embeddings_file` parameter of visualize stages has been renamed to `projected_reduced_embeddings_file` and takes an h5 file. For backwards compatibility, `projected_embeddings_file` and csv files are still accepted.
 * Added the pb_tucker model as project stage. Tucker is a contrastive learning model trained to distinguish CATH superfamilies. It consumes prottrans_bert_bfd embeddings and reduces the embedding dimensionality from 1024 to 128. See https://www.biorxiv.org/content/10.1101/2021.01.21.427551v1
 * Renamed `half_model` to `half_precision_model`

## v0.1.7

 * Added `prottrans_t5_uniref50`/`ProtTransT5UniRef50Embedder`. This version improves over T5 BFD by being finetuned on UniRef50.
 * Added a `half_model` option to both T5 models (`prottrans_t5_uniref50` and `prottrans_t5_bfd`). On the tested GPU (Quadro RTX 3000) `half_model: True` reduces memory consumption
    from 12GB to 7GB while the effect in benchmarks is negligible (±0.1 percentages points in different sets,
    generally below standard error). We therefore recommend switching to `half_model: True` for T5.
 * Added [DeepBLAST](https://github.com/flatironinstitute/deepblast) from [Protein Structural Alignments From Sequence](https://www.biorxiv.org/content/10.1101/2020.11.03.365932v1) (see example/deepblast for an example)
 * Dropped python 3.6 support and added python 3.9 support
 * Updated the docker example to cache weights

## v0.1.6

 * Updated to pytorch 1.7
 * Published the ghcr.io/bioembeddings/bio_embeddings docker image

## v0.1.5

 * Integrated [Evolutionary Scale Modeling (ESM)](https://github.com/facebookresearch/esm) from ["Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences" (Rives et al., 2019)](https://www.biorxiv.org/content/10.1101/622803v3)
 * Included [example](examples/goPredSim) to transfer GO annotations (a-la [goPredSim](https://github.com/Rostlab/goPredSim)). We also make the reference annotations and embeddings available!
 * New language models: [ESM](https://github.com/facebookresearch/esm), [PLUS](https://github.com/mswzeus/PLUS/), [CPCProt](https://github.com/amyxlu/CPCProt), [bepler](https://github.com/tbepler/protein-sequence-embedding-iclr2019) and [T5 from ProtTrans](https://github.com/agemagician/ProtTrans)
 * The documentation got a new home <https://docs.bioembeddings.com>. This includes documentation for the python API.
 * Additional pipeline and notebook examples
 * Added as `original_id` attribute to embeddings in the h5 files which contains the sequence header from the fasta file
 * Changed SeqVec to by default run a warmup so that the first embeddings don't have a random error
 * Added an `fp16` to save embeddings with half precision
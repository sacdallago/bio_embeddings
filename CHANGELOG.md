# Changelog

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
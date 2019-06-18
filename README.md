# Example README 



```python
from bio_embeddings import ElmoEmbedder, Word2VecEmbedder, FastTextEmbedder, GloveEmbedder, TransformerXLEmbedder
```

    /Users/chdallago/miniconda3/envs/allennlp/lib/python3.6/site-packages/sklearn/utils/linear_assignment_.py:21: DeprecationWarning: The linear_assignment_ module is deprecated in 0.21 and will be removed from 0.23. Use scipy.optimize.linear_sum_assignment instead.
      DeprecationWarning)



```python
sequence = "MALLHSARVLSGVASAFHPGLAAAASARASSWWAHVEMGPPDPILGVTEAYKRDTNSKKMNLGVGAYRDDNGKPYVLPSVRKAEAQIAAKGLDKEYLPIGGLAEFCRASAELALGENSEVVKSGRFVTVQTISGTGALRIGASFLQRFFKFSRDVFLPKPSWGNHTPIFRDAGMQLQSYRYYDPKTCGFDFTGALEDISKIPEQSVLLLHACAHNPTGVDPRPEQWKEIATVVKKRNLFAFFDMAYQGFASGDGDKDAWAVRHFIEQGINVCLCQSYAKNMGLYGERVGAFTVICKDADEAKRVESQLKILIRPMYSNPPIHGARIASTILTSPDLRKQWLQEVKGMADRIIGMRTQLVSNLKKEGSTHSWQHITDQIGMFCFTGLKPEQVERLTKEFSIYMTKDGRISVAGVTSGNVGYLAHAIHQVTK"
```


```python
embedders = []
embedders.append(ElmoEmbedder())
embedders.append(Word2VecEmbedder())
embedders.append(FastTextEmbedder())
embedders.append(GloveEmbedder())
embedders.append(TransformerXLEmbedder())
```

    Downloading files ELMO v1 embedder
    Downloading weights from http://maintenance.dallago.us/public/embeddings/embedding_models/seqvec/weights.hdf5
    Downloading options from http://maintenance.dallago.us/public/embeddings/embedding_models/seqvec/options.json
    Downloading subcellular location checkpoint from http://maintenance.dallago.us/public/embeddings/feature_models/seqvec/subcell_checkpoint.pt
    Downloading secondary structure checkpoint from http://maintenance.dallago.us/public/embeddings/feature_models/seqvec/secstruct_checkpoint.pt
    Downloaded files for ELMO v1 embedder
    CUDA NOT available
    Downloading files word2vec embedder
    Downloading model file from http://maintenance.dallago.us/public/embeddings/embedding_models/word2vec/word2vec.model
    Downloaded files for word2vec embedder


    /Users/chdallago/miniconda3/envs/allennlp/lib/python3.6/site-packages/smart_open/smart_open_lib.py:398: UserWarning: This function is deprecated, use smart_open.open instead. See the migration notes for details: https://github.com/RaRe-Technologies/smart_open/blob/master/README.rst#migrating-to-the-new-open-function
      'See the migration notes for details: %s' % _MIGRATION_NOTES_URL


    Downloading files fasttext embedder
    Downloading model file from http://maintenance.dallago.us/public/embeddings/embedding_models/fasttext/fasttext.model
    Downloaded files for fasttext embedder


    /Users/chdallago/miniconda3/envs/allennlp/lib/python3.6/site-packages/smart_open/smart_open_lib.py:398: UserWarning: This function is deprecated, use smart_open.open instead. See the migration notes for details: https://github.com/RaRe-Technologies/smart_open/blob/master/README.rst#migrating-to-the-new-open-function
      'See the migration notes for details: %s' % _MIGRATION_NOTES_URL


    Downloading files glove embedder
    Downloading model file from http://maintenance.dallago.us/public/embeddings/embedding_models/glove/glove.model
    Downloaded files for glove embedder
    Downloading files transformer_base embedder
    Downloading model file from http://maintenance.dallago.us/public/embeddings/embedding_models/transformerxl_base/model.pt
    Downloading vocabulary file from http://maintenance.dallago.us/public/embeddings/embedding_models/transformerxl_base/vocab.pt
    Downloaded files for transformer_base embedder


    /Users/chdallago/miniconda3/envs/allennlp/lib/python3.6/site-packages/torch/serialization.py:454: SourceChangeWarning: source code of class 'mem_transformer.MemTransformerLM' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
      warnings.warn(msg, SourceChangeWarning)
    /Users/chdallago/miniconda3/envs/allennlp/lib/python3.6/site-packages/torch/serialization.py:454: SourceChangeWarning: source code of class 'torch.nn.modules.container.ModuleList' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
      warnings.warn(msg, SourceChangeWarning)
    /Users/chdallago/miniconda3/envs/allennlp/lib/python3.6/site-packages/torch/serialization.py:454: SourceChangeWarning: source code of class 'torch.nn.modules.sparse.Embedding' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
      warnings.warn(msg, SourceChangeWarning)
    /Users/chdallago/miniconda3/envs/allennlp/lib/python3.6/site-packages/torch/serialization.py:454: SourceChangeWarning: source code of class 'torch.nn.modules.container.ParameterList' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
      warnings.warn(msg, SourceChangeWarning)
    /Users/chdallago/miniconda3/envs/allennlp/lib/python3.6/site-packages/torch/serialization.py:454: SourceChangeWarning: source code of class 'torch.nn.modules.dropout.Dropout' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
      warnings.warn(msg, SourceChangeWarning)
    /Users/chdallago/miniconda3/envs/allennlp/lib/python3.6/site-packages/torch/serialization.py:454: SourceChangeWarning: source code of class 'torch.nn.modules.linear.Linear' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
      warnings.warn(msg, SourceChangeWarning)
    /Users/chdallago/miniconda3/envs/allennlp/lib/python3.6/site-packages/torch/serialization.py:454: SourceChangeWarning: source code of class 'torch.nn.modules.activation.ReLU' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
      warnings.warn(msg, SourceChangeWarning)



```python
embeddings = []
for embedder in embedders:
    embedding = embedder.embed(sequence)
    embeddings.append(embedding)
```


```python
import numpy as np
for embedding in embeddings:
    print(np.array(embedding).shape)
```

    (3, 430, 1024)
    (430, 512)
    (430, 512)
    (430, 512)
    (430, 512)



```python

```

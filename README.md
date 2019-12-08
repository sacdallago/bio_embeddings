# This repository is a WIP
- **Goal:** a python package where sequence goes in --> embeddings come out 
- **Goal #2:** provide as many models as possible   
- **Goal #3** include feature predictors


### Progress

Embedders:   
  - [x] SeqVec v1
  - [ ] SeqVec v2
  - [ ] TransformerXL
  - [x] Fastext
  - [x] Glove
  - [x] Word2Vec

Feature extractors
  - SeqVec v1
    - [x] DSSP8
    - [x] DSSP3
    - [x] Disorder
    - [x] Subcell loc
    - [x] Membrane boundness
  - SeqVec v2
  - TransformerXL
  - Fastext
  - Glove
  - Word2Vec
  
  
  ### Todo
  
  - Decouple embedders from feature extractors
  - Add more embedders
  
  ### Wanna use it now?
  
Use the `notebooks` folder, that will always include the latest version of the src. Note: although this is in alpha, we will try to keep the API consistent.

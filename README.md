# MediX
### Translate expert level medical texts into simple language using Seq2Seq Transformer
## Problem Statement : 
- Medical texts are often written in complex language that limits accessibility for patients and non-experts. This project aims to build a transformer-based sequence-to-sequence model that translates expert-level medical text into simplified, patient-friendly language while preserving clinical meaning. 
  
## Dataset :
- Med-EASi (Medical dataset for Elaborative and Abstractive Simplification), a uniquely crowdsourced and finely annotated dataset for supervised simplification of short medical texts. It contains 1979 expert-simple text pairs in medical domain, spanning a total of 4478 UMLS concepts across all text pairs. The dataset is annotated with four textual transformations: replacement, elaboration, insertion and deletion.

## Model : 
- T5-small (for rapid experimentation and debugging)
- T5-base (for final training and evaluation)

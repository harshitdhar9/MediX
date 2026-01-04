# MediX
### Translate expert level medical texts into simple language using Seq2Seq Transformer
## Problem Statement : 
- Medical texts are often written in complex language that limits accessibility for patients and non-experts. This project aims to build a transformer-based sequence-to-sequence model that translates expert-level medical text into simplified, patient-friendly language while preserving clinical meaning. 
  
## Dataset :
- Med-EASi (Medical dataset for Elaborative and Abstractive Simplification), a uniquely crowdsourced and finely annotated dataset for supervised simplification of short medical texts. It contains 1979 expert-simple text pairs in medical domain, spanning a total of 4478 UMLS concepts across all text pairs. The dataset is annotated with four textual transformations: replacement, elaboration, insertion and deletion.

## Model : 
- T5-small (for rapid experimentation and debugging)
- T5-base (for final training and evaluation)

## Related Work :
- Prior work has explored improving medical text simplification by explicitly optimizing for readability using specialized loss functions and decoding strategies. In contrast, MediX focuses on building a clean, production-oriented sequence-to-sequence baseline using pretrained transformers, with an emphasis on model understanding, training stability, and deployment readiness.
- https://aclanthology.org/2023.findings-emnlp.322.pdf

## Setup :
Follow the steps below to set up the environment, prepare the dataset, and train the models.
### Create and activate virtual environment

```bash

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cd src
python data.py

python train_t5_small.py
python train_t5_base.py

```

## Training and Validation Loss Analysis :

### T5-small Results

| Epoch | Train Loss | Val Loss |
|------:|-----------:|---------:|
| 1 | 2.5219 | 1.4089 |
| 2 | 1.2691 | 1.2799 |
| 3 | 1.0649 | 1.1954 |
| 4 | 0.9627 | 1.1645 |
| 5 | 0.9194 | 1.1480 |
| 6 | 0.9069 | **1.1250** |

**Best checkpoint:** Epoch 6

---

### T5-base Results

| Epoch | Train Loss | Val Loss |
|------:|-----------:|---------:|
| 1 | 1.4091 | 1.1917 |
| 2 | 0.8882 | 1.1215 |
| 3 | 0.8141 | **1.0897** |
| 4 | 0.7590 | 1.0963 |
| 5 | 0.6910 | 1.1051 |
| 6 | 0.6263 | 1.1098 |

**Best checkpoint:** Epoch 3

---

### Model Comparison

| Model | Best Epoch | Best Val Loss |
|------|-----------:|--------------:|
| T5-small | 6 | 1.1250 |
| T5-base | 3 | **1.0897** |

---
### T5-small:

<img width="609" height="457" alt="Screenshot 2026-01-04 at 2 04 34 PM" src="https://github.com/user-attachments/assets/4117cc87-e3bf-4b79-84ba-10c767207ca1" />

### T5-base:

<img width="600" height="465" alt="Screenshot 2026-01-04 at 2 04 13 PM" src="https://github.com/user-attachments/assets/e9f3425d-03a8-4594-b9b8-d7d4de88b7c8" />



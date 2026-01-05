import torch 
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dataset import MedEasiDataset
from torch.utils.data import DataLoader
import textstat
from tqdm import tqdm

MODEL_PATH = "src/checkpoints1/medix-t5-small"
TEST_CSV = "dataset/csv_data_processed/expert_simple_test.csv"
BATCH_SIZE = 4
MAX_LENGTH = 128

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
model.eval()
test_dataset = MedEasiDataset(TEST_CSV)
results = []

for sample in tqdm(test_dataset):
    expert_text = sample["source_text"]
    reference_simple = sample["target_text"]

    inputs = tokenizer(
        expert_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=96,
            do_sample=True,
            temperature=1.0,
            top_p=0.85,
            repetition_penalty=1.4,
            no_repeat_ngram_size=4
        )

    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    fk_grade = textstat.flesch_kincaid_grade(generated_text)
    ari_score = textstat.automated_readability_index(generated_text)

    results.append({
        "expert_text": expert_text,
        "generated_simple": generated_text,
        "reference_simple": reference_simple,
        "fk_grade": fk_grade,
        "ari": ari_score
    })

df = pd.DataFrame(results)
df.to_csv("test_results_t5_small.csv", index=False)

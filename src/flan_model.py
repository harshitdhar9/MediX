import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
import textstat
from tqdm import tqdm
import re

INPUT_CSV = "src/test_results_t5_small.csv"
OUTPUT_CSV = "test_results_t5_small_flan_fk.csv"

MODEL_NAME = "google/flan-t5-small"

MAX_INPUT_LENGTH = 256
MAX_OUTPUT_LENGTH = 80   

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()

df = pd.read_csv(INPUT_CSV)

results = []

def fk_postprocess(text: str) -> str:
    text = re.sub(r", which ", ". This ", text)
    text = re.sub(r", and ", ". ", text)
    text = re.sub(r";", ".", text)
    return text

for _, row in tqdm(df.iterrows(), total=len(df)):
    expert_text = row["expert_text"]
    t5_text = row["generated_simple"]

    prompt = (
        "Rewrite this medical text so that it can be understood by a middle school student. "
        "Use very short sentences. "
        "Use simple, common words. "
        "Avoid medical jargon when possible. "
        "Do not add new information.\n\n"
        f"{t5_text}"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_OUTPUT_LENGTH,
            do_sample=True,
            temperature=1.0,
            top_p=0.85,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
        )

    flan_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    flan_text = fk_postprocess(flan_text)

    fk_flan = textstat.flesch_kincaid_grade(flan_text)
    ari_flan = textstat.automated_readability_index(flan_text)

    results.append({
        "expert_text": expert_text,
        "t5_generated_simple": t5_text,
        "flan_generated_simple": flan_text,
        "fk_flan": fk_flan,
        "ari_flan": ari_flan
    })

out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_CSV, index=False)

print(f"Results saved to: {OUTPUT_CSV}")

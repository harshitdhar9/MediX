#Python file for testing other files and their functions for smooth functioning and understanding

"""
from dataset import MedEasiDataset
from tokenizer import TokenizerT5

dataset=MedEasiDataset("/Users/harshitdhar/Downloads/MediX/dataset/csv_data_processed/expert_simple_train.csv")
print("Dataset length",len(dataset))
print("Sample")
print(dataset[1])

tokenizer_t5_small=TokenizerT5(model_name="google-t5/t5-small",max_length=128)

batch_samples=[dataset[i] for i in range(2)]
tokens=tokenizer_t5_small.collate_fn(batch_samples)

print("Tokenizer output keys:", tokens.keys())
print("input_ids shape:", tokens["input_ids"].shape)
print("attention_mask shape:", tokens["attention_mask"].shape)
print("labels shape:", tokens["labels"].shape)
"""
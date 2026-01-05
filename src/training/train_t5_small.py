#Model training
import os
import json

import torch
from torch.utils.data import DataLoader
from dataset import MedEasiDataset
from tokenizer import TokenizerT5
from transformers import T5ForConditionalGeneration
from torch.optim import AdamW

print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())

def evaluate(model,loader,device):
    model.eval()
    total_loss=0.0
    with torch.no_grad():
        for i,batch in enumerate(loader):
            batch={k: v.to(device) for k,v in batch.items()}
            outputs=model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss=outputs.loss
            total_loss+=loss.item()
    avg_loss=total_loss/len(loader)
    model.train()
    return avg_loss

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    dataset=MedEasiDataset("dataset/csv_data_processed/expert_simple_train.csv")
    val_dataset=MedEasiDataset("dataset/csv_data_processed/expert_simple_val.csv")
    
    tokenizer=TokenizerT5(model_name="google-t5/t5-small",
                          max_length=128)

    loader=DataLoader(dataset,
                      batch_size=8,
                      shuffle=True,
                      collate_fn=tokenizer.collate_fn)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,   
        collate_fn=tokenizer.collate_fn
    )

    model=T5ForConditionalGeneration.from_pretrained("google-t5/t5-small").to(device)
    model.train()

    optimizer=AdamW(model.parameters(), lr=5e-5)
    epochs=6

    best_val_loss = float("inf")

    history = {
        "train_loss": [],
        "val_loss": []
    }

    save_dir = "checkpoints1/medix-t5-small"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        total_loss=0.0
        for i,batch in enumerate(loader):
            batch={k: v.to(device) for k,v in batch.items()}
            outputs=model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss=outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss+=loss.item()
            
        avg_loss=total_loss/len(loader)
        val_loss=evaluate(model,val_loader,device)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        print(f"\nEpoch {epoch+1} completed | Avg loss: {avg_loss:.4f}\n")
        print(f"Val loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saving best model...")

            model.save_pretrained(save_dir)
            tokenizer.tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    print("Training complete.")

    """
    batch=next(iter(loader))

    outputs=model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"]
    )

    print(outputs.loss.item())
    """

if __name__=="__main__":
    train()
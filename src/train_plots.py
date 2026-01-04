"""
This script visualizes the training and validation loss of a model.
"""

import json
import matplotlib.pyplot as plt

with open("checkpoints1/medix-t5-small/training_history.json") as f:   #Replace with the checkpoint path you want to visualise
    history = json.load(f)

plt.plot(history["train_loss"], label="Train")
plt.plot(history["val_loss"], label="Validation")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.show()

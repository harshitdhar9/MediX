#Dataset Class

import torch
from torch.utils.data import Dataset
import pandas as pd

class MedEasiDataset(Dataset):

    def __init__(self,csv_path):
        self.df=pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        return {
            "source_text":self.df.iloc[idx]["Expert"],
            "target_text":self.df.iloc[idx]["Simple"]
        }

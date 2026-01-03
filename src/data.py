#Loading dataset from Hugging Face

import pandas as pd
import os
from datasets import load_dataset

df=load_dataset("cbasu/Med-EASi")

train_data=pd.DataFrame(df['train'])
test_data=pd.DataFrame(df['test'])
val_data=pd.DataFrame(df['validation'])

train_data.to_csv("csv_data_raw/train_med_easi.csv",index=False)
test_data.to_csv("csv_data_raw/test_med_easi.csv",index=False)
val_data.to_csv("csv_data_raw/val_med_easi.csv",index=False)

df=pd.read_csv('csv_data_raw/train_med_easi.csv')
df1=pd.read_csv('csv_data_raw/val_med_easi.csv')
df2=pd.read_csv('csv_data_raw/test_med_easi.csv')

text_train=df[["Expert","Simple"]]
text_val=df1[["Expert","Simple"]]
text_test=df2[["Expert","Simple"]]

print(text_train.head())
print(text_train.shape)

text_train.to_csv("csv_data_processed/expert_simple_train.csv", index=False)
text_val.to_csv("csv_data_processed/expert_simple_val.csv", index=False)
text_test.to_csv("csv_data_processed/expert_simple_test.csv", index=False)

from transformers import AutoTokenizer

"""
Function to process samples as batches ,  It is more efficient to dynamically pad the sentences to the longest length 
in a batch during collation, instead of padding the whole dataset to the maximum length.
"""

class TokenizerT5:
    def __init__(self,model_name:str="google-t5/t5-small",max_length:int=512):
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        self.max_length=max_length

    def collate_fn(self,batch):
        source_texts=[k['source_text'] for k in batch]
        target_texts=[k["target_text"] for k in batch]

        tokenized=self.tokenizer(
            source_texts,
            text_target=target_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return tokenized

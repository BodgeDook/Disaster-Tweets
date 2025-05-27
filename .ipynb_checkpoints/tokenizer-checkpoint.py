import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
import torch

import re
import string


def clean_text(text):
    text = text.lower()

    text = re.sub(r"http\S+|www\S+|https\S+", " <URL> ", text)

    text = re.sub(r"@\w+", " <USER> ", text)

    text = re.sub(r"#(\w+)", r" \1", text)

    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"&\w+;", '', text)

    text = re.sub(r'[^\x20-\x7E]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

def tokenize(dataset_path, val=False):
    df = pd.read_csv(dataset_path)
    df["text"] = df["text"].apply(clean_text)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    encodings = tokenizer(
        df["text"].tolist(),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    tensor_dataset = None

    if not val:
        labels = torch.tensor(df["target"].tolist())

        dataset = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels
        }

        for k, v in dataset.items():
            print(f"Dataset tokenization info:\n{k}: {v.shape}")

        tensor_dataset = TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            labels
        )
    else:
        dataset = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }

        for k, v in dataset.items():
            print(f"Dataset tokenization info:\n{k}: {v.shape}")

        tensor_dataset = TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
        )

    return tensor_dataset

def collate_fn(batch, val=False):
    input_ids  = torch.stack([b[0] for b in batch])
    masks      = torch.stack([b[1] for b in batch])

    if not val:
        labels = torch.tensor([b[2] for b in batch])
        return {"input_ids": input_ids,
                "attention_mask": masks,
                "labels": labels}
    
    return {"input_ids": input_ids,
            "attention_mask": masks}

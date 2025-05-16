import pandas as pd
from transformers import BertTokenizer
import torch

import re
import string


def clean_text(text):
    text = text.lower()

    # Replace URLs with token
    text = re.sub(r"http\S+|www\S+|https\S+", " <URL> ", text)

    # Remove mentions
    text = re.sub(r"@\w+", " <USER> ", text)

    # Convert hashtags like '#happy' → 'happy'
    text = re.sub(r"#(\w+)", r" \1", text)

    # Normalize HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"&\w+;", '', text)

    # Remove non-printable/control characters
    text = re.sub(r'[^\x20-\x7E]', '', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


df = pd.read_csv("train_modified.csv")
df["text"] = df["text"].apply(clean_text)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # или 'bert-base-multilingual-cased'

encodings = tokenizer(
    df["text"].tolist(),
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

labels = torch.tensor(df["target"].tolist())

dataset = {
    "input_ids": encodings["input_ids"],
    "attention_mask": encodings["attention_mask"],
    "labels": labels
}

for k, v in dataset.items():
    print(f"INFO:\n{k}: {v.shape}")

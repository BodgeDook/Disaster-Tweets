import os
import json
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

def create_output_dir(base: str, name: str):
    path = os.path.join(base, name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "model"), exist_ok=True)
    return path

def save_json(d: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)

def to_submission_format(input_file, output_file):
    df = pd.read_csv(input_file)
    
    num_rows = len(df)
    print(f"Число строк в таблице: {num_rows}")
    
    df = df.drop(['text', 'pred_prob'], axis=1)
    
    df = df.rename(columns={'pred_label': 'target'})
    
    df.to_csv(output_file, index=False)
    print(f"Обработанная таблица сохранена в файл: {output_file}")



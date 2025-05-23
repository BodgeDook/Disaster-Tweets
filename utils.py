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

def get_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def plot_metrics(history: dict, output_path: str, title: str):
    """
    history: словарь вида {"epoch": [...], "train_loss": [...], "eval_f1": [...]} или без "eval_f1"
    """
    epochs = history.get("epoch", [])
    plt.figure()
    if "train_loss" in history and history["train_loss"]:
        plt.plot(epochs, history["train_loss"], label="Train Loss")

    eval_f1 = history.get("eval_f1", [])
    if eval_f1:
        if len(eval_f1) == len(epochs):
            plt.plot(epochs, eval_f1, label="Eval F1")
        else:
            plt.plot(list(range(1, len(eval_f1) + 1)), eval_f1, label="Eval F1")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.savefig(output_path)
    plt.close()

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



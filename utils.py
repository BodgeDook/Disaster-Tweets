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

def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def plot_metrics(history: dict, output_path: str, title: str):
    """
    history: словарь вида {"epoch": [...], "train_loss": [...], "eval_f1": [...]}
    """
    epochs = history["epoch"]
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    if "eval_f1" in history:
        plt.plot(epochs, history["eval_f1"], label="Eval F1")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def save_json(d: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)

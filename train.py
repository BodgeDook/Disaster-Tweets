import argparse
import time
import os
import pandas as pd
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from utils import create_output_dir, save_json
from tokenizer import tokenize, collate_fn
import torch
from functools import partial
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def get_train_metrics(train_csv, exp_dir):
    df_true = pd.read_csv(train_csv)["target"].tolist()
    log_path = os.path.join(exp_dir, "log_history.csv")
    df_logs = pd.read_csv(log_path)
    plt.figure()
    plt.plot(df_logs["epoch"], df_logs["loss"])
    plt.savefig(os.path.join(exp_dir, "train_loss.png"))
    metrics = {"epoch": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
    for epoch in df_logs["epoch"].unique():
        preds_path = os.path.join(exp_dir, f"predictions_epoch_{int(epoch)}.csv")
        df_pred = pd.read_csv(preds_path)
        preds = df_pred["pred_label"].tolist()
        metrics["epoch"].append(epoch)
        metrics["accuracy"].append(accuracy_score(df_true, preds))
        metrics["precision"].append(precision_score(df_true, preds))
        metrics["recall"].append(recall_score(df_true, preds))
        metrics["f1"].append(f1_score(df_true, preds))
    for name in ["accuracy", "precision", "recall", "f1"]:
        plt.figure()
        plt.plot(metrics["epoch"], metrics[name])
        plt.savefig(os.path.join(exp_dir, f"{name}.png"))
    save_json(metrics, os.path.join(exp_dir, "train_metrics.json"))

class PredictCallback(TrainerCallback):
    def __init__(self, train_csv, train_ds, exp_dir):
        self.train_csv = train_csv
        self.train_ds = train_ds
        self.exp_dir = exp_dir
        self.trainer = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            with open(os.path.join(self.exp_dir, "log_history.csv"), "a") as f:
                f.write(f"{state.epoch},{logs['loss']}\n")
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        output = self.trainer.predict(self.train_ds)
        preds = output.predictions.argmax(-1)
        df = pd.read_csv(self.train_csv)
        df["pred_label"] = preds
        df.to_csv(os.path.join(self.exp_dir, f"predictions_epoch_{int(state.epoch)}.csv"), index=False)
        return control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=time.strftime("run_%Y%m%d_%H%M%S"))
    parser.add_argument("--train_csv", type=str, default="data/train_modified.csv")
    parser.add_argument("--model_name", type=str, default="bhadresh-savani/distilbert-base-uncased-emotion")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_base", type=str, default="outputs")
    args = parser.parse_args()

    out_dir = create_output_dir(args.output_base, args.exp_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, ignore_mismatched_sizes=True
    )
    train_ds = tokenize(args.train_csv, val=False)
    train_collate = partial(collate_fn, val=False)
    predict_cb = PredictCallback(args.train_csv, train_ds, out_dir)
    training_args = TrainingArguments(
        output_dir=f"{out_dir}/model",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        optim="adamw_torch",
        lr_scheduler_type="cosine_with_restarts"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=train_collate,
        compute_metrics=None,
        callbacks=[predict_cb]
    )
    # Подготовка лог-файла
    with open(os.path.join(out_dir, "log_history.csv"), "w") as f:
        f.write("epoch,loss\n")
    # Связывание trainer для колбэка
    predict_cb.trainer = trainer

    train_result = trainer.train()
    save_json(train_result.metrics, os.path.join(out_dir, "metrics_train.json"))
    trainer.save_model(os.path.join(out_dir, "model"))
    get_train_metrics(args.train_csv, out_dir)
    print(f"Training has finished. Check the results here: {out_dir}")

if __name__ == "__main__":
    main()

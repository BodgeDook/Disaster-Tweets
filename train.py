import argparse
import time
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)
from utils import (create_output_dir, get_metrics, plot_metrics, save_json)
from tokenizer import (tokenize, collate_fn)
import torch
from torch import nn
from functools import partial

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str,
                        default=time.strftime("run_%Y%m%d_%H%M%S"),
                        help="Experiment name folder")
    parser.add_argument("--train_csv", type=str,
                        default="data/train_modified.csv")
    parser.add_argument("--model_name", type=str,
                        default="bhadresh-savani/distilbert-base-uncased-emotion")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_base", type=str, default="outputs")
    args = parser.parse_args()

    out_dir = create_output_dir(args.output_base, args.exp_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    train_ds = tokenize(args.train_csv, val=False)

    train_collate = partial(collate_fn, val=False)

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
        compute_metrics=get_metrics,
    )

    train_result = trainer.train()
    save_json(train_result.metrics, f"{out_dir}/metrics_train.json")

    trainer.save_model(f"{out_dir}/model")

    # hist = {"epoch": [], "train_loss": [], "eval_f1": []}
    # for log in trainer.state.log_history:
    #     if "epoch" in log:
    #         if "loss" in log:
    #             hist["epoch"].append(log["epoch"])
    #             hist["train_loss"].append(log["loss"])
    #         if "eval_f1" in log:
    #             hist.setdefault("eval_f1", []).append(log["eval_f1"])

    # plot_metrics(hist, f"{out_dir}/plot_train.png", "Training Loss & F1")

    print(f"Training has finished. Check the results here: {out_dir}")

if __name__ == "__main__":
    main()

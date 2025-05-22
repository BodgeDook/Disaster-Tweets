import argparse
import pandas as pd
from transformers import (AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
from utils import (compute_metrics, plot_metrics, save_json)
from tokenizer import (tokenize)
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment folder name (defined in train.py)")
    parser.add_argument("--test_csv", type=str,
                        default="data/test_modified.csv")
    args = parser.parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(f"{args.exp_dir}/model")
    test_ds = tokenize(args.test_csv)

    trainer = Trainer(model=model)

    pred_output = trainer.predict(test_ds)
    metrics_val = compute_metrics(pred_output)
    save_json(metrics_val, f"{args.exp_dir}/metrics_val.json")

    preds = pred_output.predictions.argmax(-1)
    probs = torch.nn.functional.softmax(torch.tensor(pred_output.predictions), dim=-1)[:, 1].tolist()
    df = pd.read_csv(args.test_csv)
    df["pred_label"] = preds
    df["pred_prob"] = probs
    df.to_csv(f"{args.exp_dir}/predictions.csv", index=False)

    hist = {"epoch": [1], "eval_f1": [metrics_val["f1"]]}
    plot_metrics(hist, f"{args.exp_dir}/plot_val.png", "Validation F1")

    print(f"Validation has finished. Check the predictions here: {args.exp_dir}")

if __name__ == "__main__":
    main()

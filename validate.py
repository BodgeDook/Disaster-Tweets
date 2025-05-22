import argparse
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, Trainer
from utils import get_metrics, plot_metrics, save_json
from tokenizer import tokenize, collate_fn
from functools import partial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment folder name (defined in train.py)")
    parser.add_argument("--test_csv", type=str,
                        default="data/test_modified.csv")
    args = parser.parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(
        f"{args.exp_dir}/model"
    )

    test_ds = tokenize(args.test_csv, val=True)
    test_collate = partial(collate_fn, val=True)

    trainer = Trainer(
        model=model,
        eval_dataset=test_ds,
        data_collator=test_collate,
        compute_metrics=get_metrics
    )

    eval_result = trainer.evaluate()
    save_json(eval_result, f"{args.exp_dir}/metrics_val.json")

    pred_output = trainer.predict(test_ds)
    preds = pred_output.predictions.argmax(-1)
    probs = torch.softmax(
        torch.tensor(pred_output.predictions), dim=-1
    )[:, 1].tolist()

    df = pd.read_csv(args.test_csv)
    df["pred_label"] = preds
    df["pred_prob"] = probs
    df.to_csv(f"{args.exp_dir}/predictions.csv", index=False)

    # hist = {"epoch": [1], "eval_f1": [eval_result.get("eval_f1", 0.0)]}
    # plot_metrics(hist, f"{args.exp_dir}/plot_val.png", "Validation F1")

    print(f"Validation has finished. Results are in {args.exp_dir}")

if __name__ == "__main__":
    main()

import argparse
import pandas as pd
import torch
import os
from transformers import AutoModelForSequenceClassification, Trainer, TrainerCallback
from utils import save_json, to_submission_format
from tokenizer import tokenize, collate_fn
from functools import partial
import matplotlib.pyplot as plt


# class PredictCallback(TrainerCallback):
#     def __init__(self, test_csv, train_ds, exp_dir):
#         self.test_csv = test_csv
#         self.train_ds = train_ds
#         self.exp_dir = exp_dir
#         self.trainer = None

#     def on_log(self, args, state, control, logs=None, **kwargs):
#         if logs and "loss" in logs:
#             with open(os.path.join(self.exp_dir, "callback_loss_report_val.csv"), "a") as f:
#                 f.write(f"{state.epoch},{logs["loss"]}\n")
#         return control

# def get_eval_metrics(exp_dir):
#     log_path = os.path.join(exp_dir, "callback_loss_report_val.csv")
#     df_logs = pd.read_csv(log_path)
#     epoch_list = df_logs["epoch"].tolist()
#     loss_list = df_logs["loss"].tolist()
    
#     metrics = {
#         "epoch": epoch_list,
#         "loss": loss_list,
#     }
        
#     plt.figure()    
#     plt.plot(epoch_list, loss_list)
#     plt.title(f"Eval Loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.savefig(os.path.join(exp_dir, f"loss_val.png"))
        
#     save_json(metrics, os.path.join(exp_dir, "eval_metrics.json"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment folder name (defined in train.py)")
    parser.add_argument("--test_csv", type=str,
                        default="data/test_modified.csv")
    parser.add_argument("--output_base", type=str, default="outputs")
    args = parser.parse_args()

    out_dir = os.path.join(args.output_base, args.exp_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        f"{out_dir}/model"
    )

    test_ds = tokenize(args.test_csv, val=True)
    test_collate = partial(collate_fn, val=True)
    # predict_cb = PredictCallback(args.test_csv, test_ds, out_dir)

    trainer = Trainer(
        model=model,
        eval_dataset=test_ds,
        data_collator=test_collate,
        compute_metrics=None,
        # callbacks=[predict_cb]
    )

    with open(os.path.join(out_dir, "callback_loss_report_val.csv"), "w") as f:
        f.write("epoch,loss\n")

    # predict_cb.trainer = trainer

    eval_result = trainer.evaluate()
    save_json(eval_result, f"{out_dir}/processing_report_val.json")
    # get_eval_metrics(out_dir)

    pred_output = trainer.predict(test_ds)
    preds = pred_output.predictions.argmax(-1)
    probs = torch.softmax(
        torch.tensor(pred_output.predictions), dim=-1
    )[:, 1].tolist()

    df = pd.read_csv(args.test_csv)
    df["pred_label"] = preds
    df["pred_prob"] = probs
    df.to_csv(f"{out_dir}/predictions.csv", index=False)

    to_submission_format(f"{out_dir}/predictions.csv", f"{out_dir}/predict_submissions.csv")

    print(f"Validation has finished. Results are in {out_dir}")

if __name__ == "__main__":
    main()

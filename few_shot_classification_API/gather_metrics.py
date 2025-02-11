from pathlib import Path
import pandas as pd
import os

res_dir = "results directory"
providers = os.listdir(res_dir)

df = pd.DataFrame(columns=["provider", "engine", "sampler", "n_for_train", "seed", "dataset", "accuracy", "precision", "recall", "f1"])

for pr in providers:
    c = 1
    while c <= 500:
        conf_path = Path(res_dir + f"/{pr}/{c}/config")
        metr_path = Path(res_dir + f"/{pr}/{c}/metrics.csv")
        if conf_path.is_file() and metr_path.is_file():
            with open(conf_path, "r", encoding="utf-8") as cfg:
                content = cfg.read()
                current_config = {k: v for k, v in [i.split("=") for i in content.split("\n")]}
                new_row = {
                    "engine": current_config["engine"],
                    "provider": current_config["provider"],
                    "n_for_train": current_config["n_for_train"],
                    "sampler": current_config["sampler"],
                    "seed": current_config["seed"],
                    "dataset": current_config["dataset"]
                }

            current_metrics = data = pd.read_csv(metr_path, delimiter="\t", index_col=False).iloc[0].to_dict()
            new_row["accuracy"] = current_metrics["accuracy"]
            new_row["recall"] = current_metrics["recall"]
            new_row["precision"] = current_metrics["precision"]
            new_row["f1"] = current_metrics["f1"]

            df.loc[len(df)] = new_row
            c += 1
        else:
            c += 1

df.to_csv("result.csv", sep="\t", index=False)
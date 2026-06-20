#!/usr/bin/env python3
"""Aggregate per-run metrics.json into results.csv and summary.json."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "final_runs"
OUTPUT_CSV = ARTIFACTS / "results.csv"
OUTPUT_JSON = ARTIFACTS / "results.json"
SUMMARY_JSON = ROOT / "artifacts" / "summary.json"

FIELDS = [
    "method", "model_name", "task_name", "train_subset_size", "seed",
    "accuracy", "majority_baseline", "collapsed",
    "trainable_parameters", "peft_parameters", "classifier_parameters",
    "ia3_mode", "elapsed_seconds", "learning_rate", "batch_size",
    "max_length", "warmup_ratio", "max_grad_norm", "bf16",
    "num_train_epochs", "seed_semantics",
]


def main():
    rows = []
    for mf in sorted(ARTIFACTS.glob("*/metrics.json")):
        try:
            data = json.loads(mf.read_text())
            row = {f: data.get(f) for f in FIELDS}
            rows.append(row)
        except Exception as e:
            print(f"WARN: skipping {mf}: {e}")

    rows.sort(key=lambda r: (r["model_name"], r["task_name"], r["method"],
                             r["train_subset_size"], r["seed"]))

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    OUTPUT_JSON.write_text(json.dumps(rows, indent=2))

    # Summary: per method-model-task-size aggregation
    summary = defaultdict(lambda: {"accuracies": [], "collapsed_count": 0, "n_runs": 0})
    for r in rows:
        key = f"{r['method']}__{r['model_name']}__{r['task_name']}__n{r['train_subset_size']}"
        summary[key]["accuracies"].append(r["accuracy"])
        summary[key]["collapsed_count"] += int(r["collapsed"]) if r["collapsed"] else 0
        summary[key]["n_runs"] += 1

    summary_out = {}
    for key, v in summary.items():
        accs = [a for a in v["accuracies"] if a is not None]
        summary_out[key] = {
            "mean_accuracy": float(np.mean(accs)) if accs else None,
            "std_accuracy": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "n_runs": v["n_runs"],
            "collapsed_count": v["collapsed_count"],
            "collapse_rate": v["collapsed_count"] / v["n_runs"] if v["n_runs"] > 0 else 0,
        }

    SUMMARY_JSON.write_text(json.dumps(summary_out, indent=2))
    print(f"Aggregated {len(rows)} runs -> {OUTPUT_CSV}")
    print(f"Summary: {len(summary_out)} cells -> {SUMMARY_JSON}")


if __name__ == "__main__":
    main()

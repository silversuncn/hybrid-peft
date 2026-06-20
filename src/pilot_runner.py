"""Tiny pilot runner for the hybrid PEFT grid.

Grid: 8 methods x 1 model (BERT) x 2 tasks (SST-2, RTE) x 1 size (320) x 3 seeds = 48 runs
"""
import os
os.environ.update({
    "TORCHDYNAMO_DISABLE": "1",
    "TORCH_COMPILE_DISABLE": "1",
    "TORCHINDUCTOR_DISABLE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "CUDA_MODULE_LOADING": "LAZY",
})

import copy
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import torch
torch._dynamo.config.disable = True
import numpy as np
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))
from hybrid_peft import configure_method

# --- Pilot configuration (fixed, not argparse) ---
METHODS = [
    "lora", "bitfit", "ia3",
    "lora_bitfit", "lora_ia3", "bitfit_ia3", "lora_bitfit_ia3",
    "lora_param_matched_r12",
]
MODEL = "bert-base-uncased"
TASKS = ["sst2", "rte"]
SAMPLE_SIZE = 320
SEEDS = [31, 37, 41]
EPOCHS = 10
LR = 2e-4
WARMUP_RATIO = 0.1
BATCH_SIZE = 16
MAX_LENGTH = 128

TASK_TO_KEYS = {
    "sst2": ("sentence", None),
    "rte": ("sentence1", "sentence2"),
}

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "pilot_runs"
PLOG = ROOT / "pilot_progress.log"

TOTAL_RUNS = len(METHODS) * len(TASKS) * len(SEEDS)  # 48


def tokenize_dataset(raw_dataset, tokenizer, task):
    s1_key, s2_key = TASK_TO_KEYS[task]
    def tok_fn(examples):
        if s2_key is None:
            return tokenizer(examples[s1_key], truncation=True, max_length=MAX_LENGTH, padding="max_length")
        return tokenizer(examples[s1_key], examples[s2_key], truncation=True, max_length=MAX_LENGTH, padding="max_length")
    ds = raw_dataset.map(tok_fn, batched=True)
    ds = ds.rename_column("label", "labels")
    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in ds.column_names:
        cols.append("token_type_ids")
    ds.set_format("torch", columns=cols)
    return ds


def subsample(dataset, n, seed):
    actual_n = min(n, len(dataset))
    labels = np.array(dataset["labels"])
    sss = StratifiedShuffleSplit(n_splits=1, train_size=actual_n, random_state=seed)
    idx, _ = next(sss.split(np.zeros(len(labels)), labels))
    return Subset(dataset, idx.tolist()), actual_n


def majority_baseline(dataset):
    labels = np.array(dataset["labels"])
    counts = np.bincount(labels)
    return float(counts.max()) / len(labels)


def train_eval(model, train_loader, val_loader, device):
    total_steps = len(train_loader) * EPOCHS
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=int(total_steps * WARMUP_RATIO), num_training_steps=total_steps)

    model.train()
    for epoch in range(EPOCHS):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            opt.zero_grad(set_to_none=True)

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.append(logits.argmax(-1).cpu())
            labels.append(batch["labels"].cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return float((preds == labels).mean())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Pilot: {len(METHODS)} methods x 1 model x {len(TASKS)} tasks x 1 size x {len(SEEDS)} seeds = {TOTAL_RUNS} runs")
    print("=" * 80)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, local_files_only=True, dtype=torch.bfloat16)

    results = []
    ok, fail = 0, 0
    majority_baselines = {}

    for task in TASKS:
        raw = load_dataset("glue", task)
        train_ds_full = tokenize_dataset(raw["train"], tokenizer, task)
        val_ds = tokenize_dataset(raw["validation"], tokenizer, task)
        maj = majority_baseline(val_ds)
        majority_baselines[task] = maj
        print(f"\nTask: {task}, validation size: {len(val_ds)}, majority baseline: {maj:.4f}")

        for seed in SEEDS:
            train_subset, actual_n = subsample(train_ds_full, SAMPLE_SIZE, seed)
            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

            for method in METHODS:
                run_name = f"{method}__{MODEL}__{task}__n{SAMPLE_SIZE}__ep{EPOCHS}__s{seed}"
                run_dir = ARTIFACTS / run_name
                run_dir.mkdir(parents=True, exist_ok=True)

                t0 = time.time()
                try:
                    model_copy = copy.deepcopy(base_model)
                    model, info = configure_method(model_copy, method, MODEL)
                    model = model.to(device)

                    accuracy = train_eval(model, train_loader, val_loader, device)
                    elapsed = time.time() - t0
                    collapsed = accuracy < maj

                    record = {
                        "method": method, "model_name": MODEL, "task_name": task,
                        "train_subset_size": actual_n, "num_train_epochs": EPOCHS,
                        "seed": seed, "accuracy": accuracy,
                        "majority_baseline": maj, "collapsed": collapsed,
                        "learning_rate": LR, "batch_size": BATCH_SIZE,
                        "max_length": MAX_LENGTH, "warmup_ratio": WARMUP_RATIO,
                        "bf16": True, "elapsed_seconds": round(elapsed, 1),
                        "trainable_parameters": info["trainable_params"],
                        "peft_parameters": info["peft_params"],
                        "classifier_parameters": info["classifier_params"],
                        "ia3_mode": info["ia3_mode"],
                    }
                    (run_dir / "metrics.json").write_text(json.dumps(record, indent=2))
                    results.append(record)
                    ok += 1

                    flag = " COLLAPSE" if collapsed else ""
                    ts = time.strftime("%H:%M:%S")
                    line = f"{ts} | OK   | {method:22s} | {task:5s} | s={seed} | acc={accuracy:.4f} | {elapsed:.0f}s{flag}"
                    print(f"  [{ok+fail}/{TOTAL_RUNS}] {line}")
                    with open(PLOG, "a") as f:
                        f.write(line + "\n")

                except Exception as e:
                    elapsed = time.time() - t0
                    fail += 1
                    line = f"{time.strftime('%H:%M:%S')} | FAIL | {method:22s} | {task:5s} | s={seed} | {e}"
                    print(f"  [{ok+fail}/{TOTAL_RUNS}] {line}")
                    with open(PLOG, "a") as f:
                        f.write(line + "\n")
                    traceback.print_exc()

                finally:
                    if "model" in dir():
                        del model
                    if "model_copy" in dir():
                        del model_copy
                    torch.cuda.empty_cache()

            # GC between seed groups
            gc.collect()

        del train_ds_full, val_ds
        gc.collect()

    # --- Summary ---
    print("\n" + "=" * 80)
    print("PILOT SUMMARY")
    print("=" * 80)
    print(f"\nMajority baselines: {majority_baselines}")
    print(f"Runs: {ok} OK, {fail} FAIL out of {TOTAL_RUNS}")

    print("\n%-25s %8s %8s %8s %10s" % ("Method", "SST-2", "RTE", "Mean", "Collapses"))
    print("-" * 65)
    for method in METHODS:
        method_results = [r for r in results if r["method"] == method]
        sst2_acc = np.mean([r["accuracy"] for r in method_results if r["task_name"] == "sst2"])
        rte_acc = np.mean([r["accuracy"] for r in method_results if r["task_name"] == "rte"])
        mean_acc = np.mean([r["accuracy"] for r in method_results])
        collapses = sum(1 for r in method_results if r["collapsed"])
        print("%-25s %8.4f %8.4f %8.4f %10d/6" % (method, sst2_acc, rte_acc, mean_acc, collapses))

    # Gate check
    print("\n" + "=" * 80)
    print("GATE EVALUATION")
    print("=" * 80)
    gate_pass = True
    for method in METHODS:
        method_results = [r for r in results if r["method"] == method]
        collapse_rate = sum(1 for r in method_results if r["collapsed"]) / len(method_results) if method_results else 0
        if collapse_rate > 0.5:
            print(f"  FAIL: {method} collapse rate = {collapse_rate:.0%} (>50%)")
            gate_pass = False

    if fail > 0:
        print(f"  FAIL: {fail} implementation errors")
        gate_pass = False

    accs = [r["accuracy"] for r in results]
    if accs and (max(accs) - min(accs)) < 0.01:
        print(f"  WARNING: very low differentiation (range = {max(accs)-min(accs):.4f})")

    if gate_pass:
        print("\n  GATE PASSED: Phase 2.5 complete, ready for full grid")
    else:
        print("\n  GATE FAILED: investigate before full grid")

    print("\nPILOT COMPLETE")
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()

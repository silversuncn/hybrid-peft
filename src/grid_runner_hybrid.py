"""Hybrid PEFT full grid runner with strict seed control.

8 methods x 2 models x 4 tasks x 3 sample sizes x 3 seeds = 576 runs.
Each run is fully reproducible: set_all_seeds() controls model init,
adapter init, classifier head init, and DataLoader shuffle order.
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

import argparse
import copy
import fcntl
import gc
import json
import random
import sys
import time
import traceback
from itertools import product
from pathlib import Path

import torch
torch._dynamo.config.disable = True
import numpy as np
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))
from hybrid_peft import configure_method

# --- Configuration ---
METHODS = [
    "lora", "bitfit", "ia3",
    "lora_bitfit", "lora_ia3", "bitfit_ia3", "lora_bitfit_ia3",
    "lora_param_matched_r12",
]
MODELS = ["bert-base-uncased", "roberta-base"]
TASKS = ["sst2", "mrpc", "qnli", "rte"]
SAMPLE_SIZES = [80, 320, 1280]
SEEDS = [31, 37, 41, 43, 47, 53]
EPOCHS = 10
LR = 2e-4
WARMUP_RATIO = 0.1
BATCH_SIZE = 16
MAX_LENGTH = 128

TASK_TO_KEYS = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
}

ROOT = Path(__file__).resolve().parents[1]
PLOG = ROOT / "progress.log"
LLOG = ROOT / "pipeline.log"
ARTIFACTS = ROOT / "artifacts" / "final_runs"


def set_all_seeds(seed):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log(m, level="INFO"):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    s = f"[{level}] {ts} {m}\n"
    print(s.strip(), flush=True)
    with open(LLOG, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(s)
        fcntl.flock(f, fcntl.LOCK_UN)


def logp(method, model_name, task, n, seed, acc, status, elapsed):
    ts = time.strftime("%H:%M:%S")
    s = (f"{ts} | {status:4s} | {method:22s} | {model_name:20s} | "
         f"{task:5s} | n={n:5d} | s={seed:2d} | acc={acc:.4f} | {elapsed:.0f}s\n")
    with open(PLOG, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(s)
        fcntl.flock(f, fcntl.LOCK_UN)


def make_run_dir(method, model_name, task, n, seed):
    safe_model = model_name.replace("/", "-")
    name = f"{method}__{safe_model}__{task}__n{n}__ep{EPOCHS}__s{seed}"
    return ARTIFACTS / name


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


def reinit_classifier(model):
    """Re-initialize classifier head under current RNG state."""
    for name, param in model.named_parameters():
        if "classifier" in name:
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)


def train_eval(model, train_loader, val_loader, device):
    total_steps = len(train_loader) * EPOCHS
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps)

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
    preds, labels_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.append(logits.argmax(-1).cpu())
            labels_list.append(batch["labels"].cpu())
    preds = torch.cat(preds).numpy()
    labels_arr = torch.cat(labels_list).numpy()
    return float((preds == labels_arr).mean())


def run_single(method, model_name, task, n, seed, base_model, tokenizer,
               train_ds_full, val_ds, maj_baseline, device):
    run_dir = make_run_dir(method, model_name, task, n, seed)
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"
    if metrics_path.exists() and config_path.exists():
        return None  # skip completed (both files present)

    run_dir.mkdir(parents=True, exist_ok=True)

    # Strict seed control: set before ALL random operations
    set_all_seeds(seed)

    # Subsample training data (seed-controlled via StratifiedShuffleSplit)
    train_subset, actual_n = subsample(train_ds_full, n, seed)

    # DataLoader with explicit generator for shuffle reproducibility
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

    # Model init under seed control
    set_all_seeds(seed)
    model_copy = copy.deepcopy(base_model)
    reinit_classifier(model_copy)

    # Configure PEFT method (LoRA/IA3 init under same seed)
    model, info = configure_method(model_copy, method, model_name)
    model = model.to(device)

    t0 = time.time()
    accuracy = train_eval(model, train_loader, val_loader, device)
    elapsed = time.time() - t0

    collapsed = accuracy < maj_baseline

    record = {
        "method": method,
        "model_name": model_name,
        "task_name": task,
        "train_subset_size": actual_n,
        "num_train_epochs": EPOCHS,
        "seed": seed,
        "seed_semantics": "controls data_subsample + model_init + adapter_init + classifier_init + dataloader_shuffle",
        "accuracy": accuracy,
        "majority_baseline": maj_baseline,
        "collapsed": collapsed,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "warmup_ratio": WARMUP_RATIO,
        "max_grad_norm": 1.0,
        "bf16": True,
        "elapsed_seconds": round(elapsed, 1),
        "trainable_parameters": info["trainable_params"],
        "peft_parameters": info["peft_params"],
        "classifier_parameters": info["classifier_params"],
        "ia3_mode": info["ia3_mode"],
    }

    (run_dir / "metrics.json").write_text(json.dumps(record, indent=2))
    (run_dir / "config.json").write_text(json.dumps(record, indent=2))

    del model, model_copy
    torch.cuda.empty_cache()
    return accuracy, elapsed, collapsed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--sample_sizes", nargs="+", type=int, default=SAMPLE_SIZES)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--repro_smoke", action="store_true",
                   help="Run 8-method reproducibility smoke (BERT x SST-2 x n=320 x seed=31)")
    return p.parse_args()


def repro_smoke(device):
    """Run 8 methods twice with identical config, verify exact match."""
    print("=" * 80)
    print("REPRODUCIBILITY SMOKE TEST")
    print("Running 8 methods x 2 passes, expecting identical results")
    print("=" * 80)

    model_name = "bert-base-uncased"
    task = "sst2"
    n = 320
    seed = 31

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, local_files_only=True, dtype=torch.bfloat16)
    raw = load_dataset("glue", task)
    train_ds = tokenize_dataset(raw["train"], tokenizer, task)
    val_ds = tokenize_dataset(raw["validation"], tokenizer, task)
    maj = majority_baseline(val_ds)

    pass_results = [[], []]
    for pass_idx in range(2):
        print(f"\n--- Pass {pass_idx + 1} ---")
        for method in METHODS:
            # Clean any cached state
            set_all_seeds(seed)
            train_subset, actual_n = subsample(train_ds, n, seed)
            g = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

            set_all_seeds(seed)
            model_copy = copy.deepcopy(base_model)
            reinit_classifier(model_copy)
            model, info = configure_method(model_copy, method, model_name)
            model = model.to(device)

            accuracy = train_eval(model, train_loader, val_loader, device)
            pass_results[pass_idx].append((method, accuracy))
            print(f"  {method:25s}: {accuracy:.6f}")

            del model, model_copy
            torch.cuda.empty_cache()
            gc.collect()

    print("\n--- Comparison ---")
    all_match = True
    for i, method in enumerate(METHODS):
        a1 = pass_results[0][i][1]
        a2 = pass_results[1][i][1]
        match = "MATCH" if a1 == a2 else "MISMATCH"
        if a1 != a2:
            all_match = False
        print(f"  {method:25s}: pass1={a1:.6f} pass2={a2:.6f} [{match}]")

    print("\n" + "=" * 80)
    if all_match:
        print("REPRODUCIBILITY CHECK PASSED: all 8 methods produce identical results across passes")
    else:
        print("REPRODUCIBILITY CHECK FAILED: some methods differ between passes")
    print("=" * 80)
    return all_match


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.repro_smoke:
        ok = repro_smoke(device)
        sys.exit(0 if ok else 1)

    # Full grid
    total = len(args.methods) * len(args.models) * len(args.tasks) * len(args.sample_sizes) * len(args.seeds)
    log(f"Device: {device}")
    log(f"Grid: {len(args.methods)} methods x {len(args.models)} models x {len(args.tasks)} tasks x {len(args.sample_sizes)} sizes x {len(args.seeds)} seeds")
    log(f"Total runs: {total}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    done, ok, fail, skip = 0, 0, 0, 0

    for model_name in args.models:
        log(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, local_files_only=True, dtype=torch.bfloat16)

        for task in args.tasks:
            log(f"Loading dataset: glue/{task}")
            raw = load_dataset("glue", task)
            train_ds_full = tokenize_dataset(raw["train"], tokenizer, task)
            val_ds = tokenize_dataset(raw["validation"], tokenizer, task)
            maj = majority_baseline(val_ds)
            log(f"  majority_baseline({task}) = {maj:.4f}")

            for n, seed, method in product(args.sample_sizes, args.seeds, args.methods):
                done += 1
                try:
                    result = run_single(
                        method, model_name, task, n, seed,
                        base_model, tokenizer, train_ds_full, val_ds, maj, device)
                    if result is None:
                        skip += 1
                        continue
                    acc, elapsed, collapsed = result
                    logp(method, model_name, task, n, seed, acc, "OK", elapsed)
                    ok += 1
                except Exception as e:
                    logp(method, model_name, task, n, seed, 0.0, "FAIL", 0)
                    log(f"FAIL [{done}/{total}]: {method}/{model_name}/{task}/n={n}/s={seed}: {e}", "ERROR")
                    traceback.print_exc()
                    fail += 1

                if done % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

            del train_ds_full, val_ds
            gc.collect()

        del base_model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    log(f"DONE: {ok} OK, {fail} FAIL, {skip} SKIP out of {total} total")


if __name__ == "__main__":
    main()

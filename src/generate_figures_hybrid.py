#!/usr/bin/env python3
"""Hybrid PEFT publication figures.

Fig 1: Accuracy heatmap (method x task, per model)
Fig 2: Composition delta vs max(component)
Fig 3: Param-matched comparison (hybrid vs LoRA-r12)
Fig 4: Sample size interaction
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "artifacts" / "final_runs" / "results.csv"
STATS_JSON = ROOT / "artifacts" / "statistical_analysis.json"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

METHODS = ["lora", "bitfit", "ia3", "lora_bitfit", "lora_ia3",
           "bitfit_ia3", "lora_bitfit_ia3", "lora_param_matched_r12"]
HYBRIDS = ["lora_bitfit", "lora_ia3", "bitfit_ia3", "lora_bitfit_ia3"]
MODELS = ["bert-base-uncased", "roberta-base"]
TASKS = ["sst2", "mrpc", "qnli", "rte"]
SIZES = [80, 320, 1280]

METHOD_LABELS = {
    "lora": "LoRA", "bitfit": "BitFit", "ia3": "(IA)³",
    "lora_bitfit": "LoRA+BitFit", "lora_ia3": "LoRA+(IA)³",
    "bitfit_ia3": "BitFit+(IA)³", "lora_bitfit_ia3": "All Three",
    "lora_param_matched_r12": "LoRA r=12\n(param-matched)",
}
TASK_LABELS = {"sst2": "SST-2", "mrpc": "MRPC", "qnli": "QNLI", "rte": "RTE"}
MODEL_LABELS = {"bert-base-uncased": "BERT", "roberta-base": "RoBERTa"}
HYBRID_LABELS = {
    "lora_bitfit": "LoRA+BitFit", "lora_ia3": "LoRA+(IA)³",
    "bitfit_ia3": "BitFit+(IA)³", "lora_bitfit_ia3": "All Three",
}
SIZE_COLORS = {80: "#e41a1c", 320: "#377eb8", 1280: "#4daf4a"}

METHOD_COLORS = {
    "lora": "#1f77b4", "bitfit": "#ff7f0e", "ia3": "#2ca02c",
    "lora_bitfit": "#d62728", "lora_ia3": "#9467bd",
    "bitfit_ia3": "#8c564b", "lora_bitfit_ia3": "#e377c2",
    "lora_param_matched_r12": "#7f7f7f",
}


def load_data():
    df = pd.read_csv(RESULTS_CSV)
    df["accuracy"] = df["accuracy"].astype(float)
    df["train_subset_size"] = df["train_subset_size"].astype(int)
    df["seed"] = df["seed"].astype(int)
    return df


def load_stats():
    return json.loads(STATS_JSON.read_text())


def fig1_heatmap(df):
    """Accuracy heatmap: 8 methods x 4 tasks, one panel per model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Mean Accuracy by Method and Task", fontsize=13, y=1.02)

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        sub = df[df["model_name"] == model]
        matrix = np.zeros((len(METHODS), len(TASKS)))
        for i, m in enumerate(METHODS):
            for j, t in enumerate(TASKS):
                accs = sub[(sub["method"] == m) & (sub["task_name"] == t)]["accuracy"]
                matrix[i, j] = accs.mean() if len(accs) > 0 else 0

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.45, vmax=0.90)
        ax.set_xticks(range(len(TASKS)))
        ax.set_xticklabels([TASK_LABELS[t] for t in TASKS], fontsize=10)
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=9)
        ax.set_title(MODEL_LABELS[model], fontsize=12)

        for i in range(len(METHODS)):
            for j in range(len(TASKS)):
                color = "white" if matrix[i, j] > 0.72 else "black"
                ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                       fontsize=8, color=color)

    fig.colorbar(im, ax=axes, shrink=0.8, label="Accuracy")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES / f"fig1_accuracy_heatmap.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Fig 1: accuracy heatmap saved")


def fig2_composition_delta(df, stats_data):
    """Delta vs max(component), faceted by task, grouped by sample size."""
    comps = [c for c in stats_data["block_b_paired_comparisons"]["comparisons"]
             if "vs_max_component" in c["comparison"]]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    fig.suptitle("Composition Effect: Hybrid - max(Component)", fontsize=13, y=1.02)

    for midx, model in enumerate(MODELS):
        for tidx, task in enumerate(TASKS):
            ax = axes[midx, tidx]
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

            x_positions = np.arange(len(HYBRIDS))
            width = 0.25

            for sidx, size in enumerate(SIZES):
                means, lows, highs, sigs = [], [], [], []
                for hybrid in HYBRIDS:
                    match = [c for c in comps if c["model"] == model and c["task"] == task
                             and c["sample_size"] == size and c["hybrid"] == hybrid]
                    if match:
                        c = match[0]
                        means.append(c["mean_delta"])
                        lows.append(c["mean_delta"] - c["ci_low"])
                        highs.append(c["ci_high"] - c["mean_delta"])
                        sigs.append(c["significant"])
                    else:
                        means.append(0)
                        lows.append(0)
                        highs.append(0)
                        sigs.append(False)

                offset = (sidx - 1) * width
                bars = ax.bar(x_positions + offset, means, width * 0.9,
                             yerr=[lows, highs], capsize=2,
                             color=SIZE_COLORS[size], alpha=0.8,
                             label=f"n={size}" if midx == 0 and tidx == 0 else "")
                for bi, sig in enumerate(sigs):
                    if sig:
                        ax.text(x_positions[bi] + offset, means[bi] + highs[bi] + 0.005,
                               "*", ha="center", fontsize=10, fontweight="bold")

            ax.set_xticks(x_positions)
            ax.set_xticklabels([HYBRID_LABELS[h] for h in HYBRIDS], fontsize=7, rotation=30, ha="right")
            ax.set_title(f"{MODEL_LABELS[model]} / {TASK_LABELS[task]}", fontsize=10)
            if tidx == 0:
                ax.set_ylabel("Delta (accuracy)", fontsize=9)

    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES / f"fig2_composition_delta.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Fig 2: composition delta saved")


def fig3_param_matched(df, stats_data):
    """Delta vs LoRA-r12, same layout as fig2."""
    comps = [c for c in stats_data["block_b_paired_comparisons"]["comparisons"]
             if "vs_lora_r12" in c["comparison"]]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    fig.suptitle("Parameter-Matched Control: Hybrid - LoRA r=12", fontsize=13, y=1.02)

    for midx, model in enumerate(MODELS):
        for tidx, task in enumerate(TASKS):
            ax = axes[midx, tidx]
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

            x_positions = np.arange(len(HYBRIDS))
            width = 0.25

            for sidx, size in enumerate(SIZES):
                means, lows, highs, sigs = [], [], [], []
                for hybrid in HYBRIDS:
                    match = [c for c in comps if c["model"] == model and c["task"] == task
                             and c["sample_size"] == size and c["hybrid"] == hybrid]
                    if match:
                        c = match[0]
                        means.append(c["mean_delta"])
                        lows.append(c["mean_delta"] - c["ci_low"])
                        highs.append(c["ci_high"] - c["mean_delta"])
                        sigs.append(c["significant"])
                    else:
                        means.append(0)
                        lows.append(0)
                        highs.append(0)
                        sigs.append(False)

                offset = (sidx - 1) * width
                bars = ax.bar(x_positions + offset, means, width * 0.9,
                             yerr=[lows, highs], capsize=2,
                             color=SIZE_COLORS[size], alpha=0.8,
                             label=f"n={size}" if midx == 0 and tidx == 0 else "")
                for bi, sig in enumerate(sigs):
                    if sig:
                        ax.text(x_positions[bi] + offset, means[bi] + highs[bi] + 0.005,
                               "*", ha="center", fontsize=10, fontweight="bold")

            ax.set_xticks(x_positions)
            ax.set_xticklabels([HYBRID_LABELS[h] for h in HYBRIDS], fontsize=7, rotation=30, ha="right")
            ax.set_title(f"{MODEL_LABELS[model]} / {TASK_LABELS[task]}", fontsize=10)
            if tidx == 0:
                ax.set_ylabel("Delta (accuracy)", fontsize=9)

    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES / f"fig3_param_matched.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Fig 3: param-matched comparison saved")


def fig4_sample_size(df):
    """Sample size interaction: accuracy vs n, lines per method, faceted by model x task."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=False)
    fig.suptitle("Accuracy vs Sample Size by Method", fontsize=13, y=1.02)

    for midx, model in enumerate(MODELS):
        for tidx, task in enumerate(TASKS):
            ax = axes[midx, tidx]
            sub = df[(df["model_name"] == model) & (df["task_name"] == task)]

            for method in METHODS:
                msub = sub[sub["method"] == method]
                means = []
                stds = []
                for size in SIZES:
                    accs = msub[msub["train_subset_size"] == size]["accuracy"]
                    means.append(accs.mean() if len(accs) > 0 else 0)
                    stds.append(accs.std(ddof=1) if len(accs) > 1 else 0)
                ax.errorbar(SIZES, means, yerr=stds, marker="o", markersize=3,
                           linewidth=1.2, capsize=2, label=METHOD_LABELS[method],
                           color=METHOD_COLORS[method])

            ax.set_xscale("log")
            ax.set_xticks(SIZES)
            ax.set_xticklabels([str(s) for s in SIZES])
            ax.set_title(f"{MODEL_LABELS[model]} / {TASK_LABELS[task]}", fontsize=10)
            if tidx == 0:
                ax.set_ylabel("Accuracy", fontsize=9)
            if midx == 1:
                ax.set_xlabel("Sample size", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=7, bbox_to_anchor=(1.12, 0.5))
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES / f"fig4_sample_size.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Fig 4: sample size interaction saved")


def main():
    df = load_data()
    stats_data = load_stats()
    print(f"Loaded {len(df)} runs")

    fig1_heatmap(df)
    fig2_composition_delta(df, stats_data)
    fig3_param_matched(df, stats_data)
    fig4_sample_size(df)

    print(f"\nAll figures saved to {FIGURES}")
    print(f"Files: {sorted(f.name for f in FIGURES.iterdir())}")


if __name__ == "__main__":
    main()

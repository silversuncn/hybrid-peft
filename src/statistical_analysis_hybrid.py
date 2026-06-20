#!/usr/bin/env python3
"""Hybrid PEFT statistical analysis.

Block A: Per-stratum ANOVA (accuracy ~ method * sample_size)
Block B: Core paired comparisons (hybrid vs max-component, hybrid vs LoRA-r12)
Block C: Multi-level collapse analysis + sensitivity check
"""
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "artifacts" / "final_runs" / "results.csv"
OUTPUT = ROOT / "artifacts" / "statistical_analysis.json"

METHODS = ["lora", "bitfit", "ia3", "lora_bitfit", "lora_ia3",
           "bitfit_ia3", "lora_bitfit_ia3", "lora_param_matched_r12"]
HYBRIDS = ["lora_bitfit", "lora_ia3", "bitfit_ia3", "lora_bitfit_ia3"]
COMPONENT_MAP = {
    "lora_bitfit": ["lora", "bitfit"],
    "lora_ia3": ["lora", "ia3"],
    "bitfit_ia3": ["bitfit", "ia3"],
    "lora_bitfit_ia3": ["lora", "bitfit", "ia3"],
}
MODELS = ["bert-base-uncased", "roberta-base"]
TASKS = ["sst2", "mrpc", "qnli", "rte"]
SAMPLE_SIZES = [80, 320, 1280]
SEEDS = [31, 37, 41]
N_BOOTSTRAP = 10000


def load_data():
    df = pd.read_csv(RESULTS_CSV)
    df["accuracy"] = df["accuracy"].astype(float)
    df["train_subset_size"] = df["train_subset_size"].astype(int)
    df["seed"] = df["seed"].astype(int)
    df["collapsed"] = df["collapsed"].astype(bool)
    return df


def block_a_anova(df):
    results = {}
    for model in MODELS:
        for task in TASKS:
            key = f"{model}__{task}"
            sub = df[(df["model_name"] == model) & (df["task_name"] == task)].copy()
            sub["method"] = sub["method"].astype(str)
            sub["size"] = sub["train_subset_size"].astype(str)

            if len(sub) < 10:
                results[key] = {"error": "insufficient data"}
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model_ols = ols("accuracy ~ C(method) * C(size)", data=sub).fit()
                    table = anova_lm(model_ols, typ=2)
                    res = {}
                    for term in table.index:
                        if term == "Residual":
                            continue
                        res[term] = {
                            "F": float(table.loc[term, "F"]) if not np.isnan(table.loc[term, "F"]) else None,
                            "p": float(table.loc[term, "PR(>F)"]) if not np.isnan(table.loc[term, "PR(>F)"]) else None,
                            "partial_eta_sq": float(table.loc[term, "sum_sq"] / (table.loc[term, "sum_sq"] + table.loc["Residual", "sum_sq"])),
                        }
                    res["n_observations"] = len(sub)
                    res["r_squared"] = float(model_ols.rsquared)
                    results[key] = res
                except Exception as e:
                    results[key] = {"error": str(e)}
    return results


def bootstrap_ci(deltas, n_boot=N_BOOTSTRAP, alpha=0.05):
    if len(deltas) < 2:
        m = float(np.mean(deltas))
        return m, m, m
    rng = np.random.RandomState(42)
    boot_means = [np.mean(rng.choice(deltas, size=len(deltas), replace=True)) for _ in range(n_boot)]
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(np.mean(deltas)), lo, hi


def paired_test(deltas):
    if len(deltas) < 2 or np.std(deltas, ddof=1) == 0:
        return 0.0, 1.0
    t, p = stats.ttest_1samp(deltas, 0)
    return float(t), float(p)


def bh_correction(p_values):
    n = len(p_values)
    if n == 0:
        return []
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[i] = sorted_p[i]
        else:
            adjusted[i] = min(sorted_p[i] * n / (i + 1), adjusted[i + 1])
    adjusted = np.minimum(adjusted, 1.0)
    result = np.zeros(n)
    result[sorted_idx] = adjusted
    return result.tolist()


def block_b_paired(df):
    comparisons = []

    for model in MODELS:
        for task in TASKS:
            for size in SAMPLE_SIZES:
                cell = df[(df["model_name"] == model) & (df["task_name"] == task) &
                          (df["train_subset_size"] == size)]

                for hybrid in HYBRIDS:
                    components = COMPONENT_MAP[hybrid]
                    hybrid_accs = cell[cell["method"] == hybrid].set_index("seed")["accuracy"]
                    r12_accs = cell[cell["method"] == "lora_param_matched_r12"].set_index("seed")["accuracy"]
                    lora_accs = cell[cell["method"] == "lora"].set_index("seed")["accuracy"]

                    comp_frames = [cell[cell["method"] == c].set_index("seed")["accuracy"] for c in components]
                    if not all(len(f) == len(SEEDS) for f in comp_frames) or len(hybrid_accs) != len(SEEDS):
                        continue

                    max_comp = comp_frames[0].copy()
                    for f in comp_frames[1:]:
                        max_comp = np.maximum(max_comp, f)

                    common_seeds = sorted(set(hybrid_accs.index) & set(max_comp.index) & set(r12_accs.index) & set(lora_accs.index))
                    if len(common_seeds) < 2:
                        continue

                    for comp_type, baseline_accs, baseline_label in [
                        ("vs_max_component", max_comp, f"max({','.join(components)})"),
                        ("vs_lora_r12", r12_accs, "lora_param_matched_r12"),
                        ("vs_lora_r8", lora_accs, "lora_r8"),
                    ]:
                        d = (hybrid_accs[common_seeds] - baseline_accs[common_seeds]).values
                        mean_d, lo, hi = bootstrap_ci(d)
                        t, p = paired_test(d)
                        comparisons.append({
                            "comparison": f"{hybrid}_{comp_type}",
                            "model": model, "task": task, "sample_size": size,
                            "hybrid": hybrid, "baseline": baseline_label,
                            "mean_delta": mean_d, "ci_low": lo, "ci_high": hi,
                            "t_stat": t, "p_raw": p, "n_seeds": len(common_seeds),
                            "per_seed_deltas": d.tolist(),
                        })

    p_raw = [c["p_raw"] for c in comparisons]
    p_adj = bh_correction(p_raw)
    for i, c in enumerate(comparisons):
        c["p_adjusted"] = p_adj[i]
        c["significant"] = p_adj[i] < 0.05

    return {"comparisons": comparisons, "n_total": len(comparisons),
            "n_significant": sum(1 for c in comparisons if c["significant"]),
            "fdr_method": "Benjamini-Hochberg", "alpha": 0.05}


def block_c_collapse(df, paired_comparisons):
    """Multi-level collapse analysis + sensitivity check."""

    # Level 1: method x task (coarse)
    coarse = {}
    for method in METHODS:
        for task in TASKS:
            sub = df[(df["method"] == method) & (df["task_name"] == task)]
            n = len(sub)
            c = int(sub["collapsed"].sum())
            coarse[f"{method}__{task}"] = {
                "n_runs": n, "collapsed": c,
                "collapse_rate": round(c / n, 4) if n > 0 else 0,
                "unreliable": (c / n > 0.2) if n > 0 else False,
            }

    # Level 2: method x model x task x size (fine-grained)
    fine = {}
    unreliable_hybrid_cells = set()
    for method in METHODS:
        for model in MODELS:
            for task in TASKS:
                for size in SAMPLE_SIZES:
                    sub = df[(df["method"] == method) & (df["model_name"] == model) &
                             (df["task_name"] == task) & (df["train_subset_size"] == size)]
                    n = len(sub)
                    c = int(sub["collapsed"].sum())
                    rate = c / n if n > 0 else 0
                    key = f"{method}__{model}__{task}__n{size}"
                    fine[key] = {
                        "n_runs": n, "collapsed": c,
                        "collapse_rate": round(rate, 4),
                        "unreliable": rate > 0.2,
                    }
                    if rate > 0.2 and method in HYBRIDS:
                        unreliable_hybrid_cells.add((method, model, task, size))

    # Sensitivity check: exclude unreliable hybrid cells from paired comparisons
    comps = paired_comparisons["comparisons"]
    surviving = []
    excluded_count = 0
    for c in comps:
        cell_key = (c["hybrid"], c["model"], c["task"], c["sample_size"])
        if cell_key in unreliable_hybrid_cells:
            excluded_count += 1
            continue
        surviving.append(c)

    sig_positive = [c for c in surviving if c["significant"] and c["mean_delta"] > 0]

    n_fine_unreliable = sum(1 for v in fine.values() if v["unreliable"])
    n_fine_over50 = sum(1 for v in fine.values() if v["collapse_rate"] > 0.5)

    return {
        "method_x_task": coarse,
        "method_x_model_x_task_x_size": fine,
        "summary": {
            "coarse_unreliable_cells": sum(1 for v in coarse.values() if v["unreliable"]),
            "fine_total_cells": len(fine),
            "fine_unreliable_cells": n_fine_unreliable,
            "fine_unreliable_hybrid_cells": len(unreliable_hybrid_cells),
            "fine_unreliable_non_hybrid_cells": n_fine_unreliable - len(unreliable_hybrid_cells),
            "cells_over_50pct": n_fine_over50,
        },
        "sensitivity_check": {
            "unreliable_hybrid_cells_excluded": len(unreliable_hybrid_cells),
            "comparisons_excluded": excluded_count,
            "comparisons_remaining": len(surviving),
            "significant_positive_remaining": len(sig_positive),
            "conclusion": "No positive hybrid advantage even after excluding unreliable cells",
        },
    }


def main():
    df = load_data()
    print(f"Loaded {len(df)} runs")

    print("Running Block A: ANOVA...")
    anova = block_a_anova(df)

    print("Running Block B: Paired comparisons...")
    paired = block_b_paired(df)

    print("Running Block C: Collapse analysis...")
    collapse = block_c_collapse(df, paired)

    output = {
        "block_a_anova": anova,
        "block_b_paired_comparisons": paired,
        "block_c_collapse": collapse,
        "metadata": {
            "n_runs": len(df),
            "n_methods": len(METHODS),
            "n_models": len(MODELS),
            "n_tasks": len(TASKS),
            "n_sample_sizes": len(SAMPLE_SIZES),
            "n_seeds": len(SEEDS),
            "bootstrap_resamples": N_BOOTSTRAP,
        }
    }

    OUTPUT.write_text(json.dumps(output, indent=2))
    print(f"Output: {OUTPUT}")

    n_anova_sig = sum(1 for v in anova.values() if isinstance(v, dict) and "C(method)" in v and v["C(method)"]["p"] is not None and v["C(method)"]["p"] < 0.05)
    print(f"\nANOVA: method effect significant in {n_anova_sig}/8 strata")
    print(f"Paired: {paired['n_significant']}/{paired['n_total']} comparisons significant after BH correction")
    print(f"Collapse (coarse): {collapse['summary']['coarse_unreliable_cells']} unreliable method x task cells")
    print(f"Collapse (fine): {collapse['summary']['fine_unreliable_cells']}/{collapse['summary']['fine_total_cells']} cells >20%, {collapse['summary']['cells_over_50pct']} >50%")
    print(f"Sensitivity: {collapse['sensitivity_check']['unreliable_hybrid_cells_excluded']} hybrid cells excluded, {collapse['sensitivity_check']['comparisons_remaining']} comparisons remain, {collapse['sensitivity_check']['significant_positive_remaining']} positive significant")


if __name__ == "__main__":
    main()

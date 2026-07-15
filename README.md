# Disentangling Capacity from Complementarity in Combined LoRA, BitFit, and (IA)³ Fine-Tuning

This repository contains the experiment code and data for the paper:

> **Disentangling Capacity from Complementarity in Combined LoRA, BitFit, and (IA)³ Fine-Tuning**
> Yaowen Sun, Xin Zhang, Ning Wu

## Overview

We test whether simple simultaneous composition of LoRA, BitFit, and (IA)³ provides statistically significant complementary benefit for low-resource text classification beyond the strongest single component or a parameter-matched LoRA control. The experiments cover:

- **Models**: BERT-base-uncased, RoBERTa-base
- **Tasks**: SST-2, MRPC, QNLI, RTE (GLUE)
- **Sample sizes**: 80, 320, 1280
- **PEFT methods**: LoRA-r8, BitFit, (IA)³, LoRA+BitFit, LoRA+(IA)³, BitFit+(IA)³, LoRA+BitFit+(IA)³, LoRA-r12 (parameter-matched control)
- **Seeds**: 31, 37, 41, 43, 47, 53
- **Total runs**: 1,152

## Repository Structure

```
├── src/
│   ├── hybrid_peft.py              # Method implementations (8 PEFT configurations)
│   ├── grid_runner_hybrid.py       # Grid search launcher
│   ├── aggregate_results_hybrid.py # Aggregate per-run metrics into results.csv
│   ├── statistical_analysis_hybrid.py # Type II ANOVA + BH-corrected paired tests
│   ├── generate_figures_hybrid.py  # Publication figure generation
│   ├── pilot_runner.py             # Pilot experiment runner
│   ├── smoke_test.py               # Environment smoke test
│   └── preflight.py                # (not included; env check utility)
├── data/
│   └── results.csv                 # Aggregated results (1,152 rows)
├── figures/                        # Publication figures (PDF)
├── LICENSE
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch 2.1+
- Transformers 4.36+
- PEFT 0.7+
- scipy, statsmodels, pandas, numpy, matplotlib, seaborn

## Usage

```bash
# Run statistical analysis on pre-computed results
python src/statistical_analysis_hybrid.py

# Generate publication figures
python src/generate_figures_hybrid.py

# Aggregate raw run metrics (requires artifacts/ directory)
python src/aggregate_results_hybrid.py
```

The `data/results.csv` file contains all 1,152 experiment runs (8 methods × 2 models × 4 tasks × 3 sizes × 6 seeds). Each row records the method, model, task, sample size, seed, accuracy, and training configuration.

## Key Findings

- ANOVA detects significant method effects in all 8 model–task strata
- **0/96** LoRA-hybrid-vs-best-component comparisons are significant positive after BH-FDR correction
- **0/96** hybrid-vs-LoRA-r12 comparisons show positive significant advantage; **15/96** are significantly negative
- **26/288** total pairwise comparisons are significant (3 positive for BitFit+(IA)³ vs its weak components; 23 negative)
- Collapse diagnostics: 28/192 fine-grained cells exceed 20% collapse rate

The supported conclusion: this grid provides no corrected statistical evidence that naive PEFT composition is a reliable substitute for controlled capacity allocation.

## Citation

```bibtex
@article{sun2026hybrid,
  title={Disentangling Capacity from Complementarity in Combined LoRA, BitFit, and (IA)³ Fine-Tuning},
  author={Sun, Yaowen and Zhang, Xin and Wu, Ning},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).

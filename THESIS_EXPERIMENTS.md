# Thesis Experiments: LGNO Coupling Analysis

This document describes experimental extensions developed for a master's thesis using the original LGNO implementation for the one-dimensional **Burgers** and **PureConvection** benchmarks.

The original paper implementation and architecture are retained. The thesis-specific scripts are in `task2_ablation/`, `task3_analysis/`, and `task4_finetuning/`. See the main [`README.md`](README.md) for the paper citation, license, and original benchmark documentation.

## Setup and required files

Run commands from the **repository root**. Install the original project and its dependencies:

```bash
pip install -e .
pip install -r requirements.txt
```

The thesis scripts require data and pretrained model checkpoints that **are not distributed in this fork**. Before running a command, place the needed files at the expected relative paths, or use an explicit file-path argument where the script supports one. The training scripts from the original project provide a starting point for creating checkpoints and datasets; equivalence with the historical thesis files is not guaranteed without matching their generation settings.

Pretrained LGNO checkpoints used by the experiments:

- `Burgers/checkpoints/burgers_hybrid_flowmap_dt.pt`
- `PureConvection/checkpoints/pureconvection_hybrid_flowmap_dt.pt`

The Task 2 one-step and Task 3 PureConvection analyses also require:

- `Burgers/dataset/eval_test_data/test_pv.pt` (Task 2 one-step)
- `PureConvection/dataset/test_pv.pt` (Task 2 one-step and Task 3)

Generated results are written under the task directories and are excluded from Git by `.gitignore`. Use **new output directories** for repeated experiments to avoid overwriting previous results.

## Task 1 — Original evaluation

Task 1 uses the original `Burgers/eval.py` and `PureConvection/eval.py`. Refer to the main README for their evaluation commands.

## Task 2 — FULL versus NO-C ablation

The **FULL** model is the original pretrained LGNO. **NO-C** uses the same weights but temporarily zeroes the multiplicative coupling channels at the mixer input. The intervention does not edit the pretrained checkpoint.

### One-step evaluation

```bash
python task2_ablation/eval_one_step.py --problem PureConvection
python task2_ablation/eval_one_step.py --problem Burgers
```

These commands use the original one-step test datasets and checkpoints listed above. They save metrics by default beneath `task2_ablation/results/`.

### Autoregressive rollout with WENO-Z references

```bash
python task2_ablation/eval_rollout.py --problem PureConvection --T 3
python task2_ablation/eval_rollout.py --problem Burgers --T 4 --seed 2
```

The evaluator reuses the original benchmark code to generate initial conditions and WENO-Z reference trajectories. It needs the pretrained checkpoint but not a stored one-step test dataset. It refuses to write into a nonempty output directory. With default settings, the Burgers example writes an archive at:

`task2_ablation/results/Burgers/rollout/T4_seed2/trajectories_and_metrics.npz`

### Burgers shock-region diagnostics

After the Burgers rollout, shock locations and near/far error statistics can be generated with:

```bash
python task2_ablation/detect_weno_shocks.py --data task2_ablation/results/Burgers/rollout/T4_seed2/trajectories_and_metrics.npz --all --out-dir task2_ablation/results/Burgers/shock_detection_weno
python task2_ablation/compare_shock_regions.py --data task2_ablation/results/Burgers/rollout/T4_seed2/trajectories_and_metrics.npz --front-dir task2_ablation/results/Burgers/shock_detection_weno --out-dir task2_ablation/results/Burgers/shock_comparison
```

The detection parameters include `--threshold-fraction` (default `0.06`), `--max-peak-fraction` (default `0.12`), and `--min-separation` (default `4`). Check the generated CSV name before proceeding to Task 3; that script expects a specific default shock-detection filename.

`eval_total_variation.py` is an **additional diagnostic** and was not a main Report 1 table result. Its input archive can be selected with `--data` and its output with `--outdir`.

## Task 3 — Branch and coupling contributions

`task3_analyze_coupling.py` inspects fixed mixer matrices and measures the actual contributions of the hidden/residual, global, local, and coupling branches in the **unmodified FULL model**.

```bash
python task3_analysis/task3_analyze_coupling.py --problem PureConvection --n-samples 50
python task3_analysis/task3_analyze_coupling.py --problem Burgers --n-samples 50
```

The PureConvection analysis requires its pretrained checkpoint and `PureConvection/dataset/test_pv.pt`.

The Burgers `--source main` analysis additionally expects the Task 2 archive and the shock-detection CSV at these specific paths:

- `task2_ablation/results/Burgers/rollout/T4_seed2/trajectories_and_metrics.npz`
- `task2_ablation/results/Burgers/shock_detection_weno/weno_fronts_all_thr0p060_peak0p120.csv`

The default output location is `task3_analysis/results/<problem>/main/`. The analysis writes `contributions_per_case.csv`, summary CSV files, plots, and `run_metadata.json` with input SHA-256 fingerprints.

The second analysis uses the first script's CSV:

```bash
python task3_analysis/task3_compare_branches.py --input task3_analysis/results/Burgers/main/contributions_per_case.csv --outdir task3_analysis/results/Burgers/paired_comparison_new
```

The optional Burgers `--source step` mode depends on a separate piecewise-constant experiment; its generation script and historical input data are **not included** in this initial release.

## Task 4A — Mixer-only NO-C fine-tuning

The historical Task 4A CPU experiments used **200 training**, **40 validation**, and **20 test initial conditions per PDE**, with **20 temporal one-step transitions per initial condition**, **20 epochs**, **batch size 8**, and **mixer-only** training. The unified training script also supports later configurations; its defaults (**160 epochs**, **batch size 64**) are **not** the historical Report 1 settings.

Historical datasets are organized under:

- `task4_finetuning/datasets/Burgers_200ic_seed20261110/`
- `task4_finetuning/datasets/PureConvection_200ic_seed20261110/`

Each directory contains `train_pv.pt`, `val_pv.pt`, and `test_pv.pt`. These files are **not distributed** with the fork.

Example invocation showing the main historical CPU settings (for Burgers):

```bash
python task4_finetuning/train_no_c.py --problem Burgers --train-file task4_finetuning/datasets/Burgers_200ic_seed20261110/train_pv.pt --val-file task4_finetuning/datasets/Burgers_200ic_seed20261110/val_pv.pt --trainable mixer-only --device cpu --epochs 20 --batch-size 8 --outdir task4_finetuning/results/Burgers/reproduction_new
```

**Important:** this is a usage example, **not a verified exact reproduction command**. Exact historical replication also requires confirmation of the original optimizer, learning-rate/scheduler settings, random seed, and update limit. Do not equate `--n-train` and `--n-val` with initial-condition counts: the script reads one-step **pairs** from the archives.

Evaluate FULL, original NO-C, and fine-tuned NO-C using a matching Task 4A test archive and the fine-tuned checkpoint:

```bash
python task4_finetuning/eval_no_c.py --problem Burgers --test-file task4_finetuning/datasets/Burgers_200ic_seed20261110/test_pv.pt --finetuned-checkpoint task4_finetuning/results/Burgers/reproduction_new/best_no_c_mixer_finetuned.pt --outdir task4_finetuning/results/Burgers/evaluation_new
```

The evaluator also accepts `--full-checkpoint` to override the pretrained model path. The historical Report 1 fine-tuned checkpoints are not included in the repository.

## Scope and attribution

Only the thesis scripts and documentation are included in these extensions. Checkpoints, datasets, generated CSV/NPZ files, plots, and experiment outputs remain excluded. An optional piecewise-constant Burgers experiment and subsequent CPU/GPU optimization studies are not part of this initial documented release.

The underlying LGNO implementation originates from the authors of **LGNO: A Local--Global Neural Operator for Hyperbolic Conservation Laws**. The original paper citation and the repository's license are retained in the main README and `LICENSE` file.

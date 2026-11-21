"""Task 2: one-step comparison of pretrained LGNO FULL and NO-C.

Run from the repository root:
    python .\task2_ablation\eval_one_step.py --problem PureConvection
    python .\task2_ablation\eval_one_step.py --problem Burgers

FULL and NO-C use the same architecture, checkpoint, and test dataset.
The `without_coupling` context manager zeros only the c^ell channels
before each mixer processes its input. No model weights are changed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ablation import without_coupling
from conslaw.checkpoints import load_hybrid_step_map_1d
from conslaw.eval_reports import dataset_error_row, save_test_metrics_csv

# Match the default precision used by the original evaluation programs.
torch.set_default_dtype(torch.float64)
#Burgers or Pureconvection choice
CONFIG = {
    "PureConvection": {
        "ckpt": ROOT / "PureConvection" / "checkpoints" / "pureconvection_hybrid_flowmap_dt.pt",
        "test": ROOT / "PureConvection" / "dataset" / "test_pv.pt",
    },
    "Burgers": {
        "ckpt": ROOT / "Burgers" / "checkpoints" / "burgers_hybrid_flowmap_dt.pt",
        "test": ROOT / "Burgers" / "dataset" / "eval_test_data" / "test_pv.pt",
    },
}


def per_example_errors(pred: np.ndarray, reference: np.ndarray):
    """Compute the same per-example metrics as conslaw.eval_reports (eps=1e-12)."""
    difference = np.abs(pred - reference)
    magnitude = np.abs(reference)
    axes = tuple(range(1, pred.ndim))
    abs_l1 = difference.mean(axis=axes)
    rel_l1 = abs_l1 / (magnitude.mean(axis=axes) + 1e-12)
    abs_linf = difference.max(axis=axes)
    rel_linf = abs_linf / (magnitude.max(axis=axes) + 1e-12)
    return {
        "abs_l1": abs_l1,
        "rel_l1": rel_l1,
        "abs_linf": abs_linf,
        "rel_linf": rel_linf,
    }


def write_csv(path: Path, columns: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=CONFIG, required=True)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    config = CONFIG[args.problem]
    outdir = args.outdir or (ROOT / "task2_ablation" / "results" / args.problem / "one_step")
    outdir.mkdir(parents=True, exist_ok=True)

    for key in ("ckpt", "test"):
        if not config[key].is_file():
            raise FileNotFoundError(f"Missing {key} file: {config[key]}")

    device = torch.device(args.device)
    # loading model
    model, ckpt = load_hybrid_step_map_1d(str(config["ckpt"]), device=device)
    model.eval()
    dt_model = float(ckpt["dt"])
    print(f"[checkpoint] {config['ckpt'].name} | dt={dt_model} | device={device}")

    #   Loading dataset (the original .pt test archive, as in the published evaluators)
    dataset = torch.load(config["test"], map_location="cpu", weights_only=False)
    x = dataset["input"].unsqueeze(-1).to(dtype=torch.float64)
    target = dataset["output"].unsqueeze(-1).to(dtype=torch.float64).numpy()
    if x.shape != target.shape:
        raise ValueError(f"Input and target shapes differ: {x.shape}, {target.shape}")
    n = x.shape[0]

    metadata = dataset.get("meta", {})
    if "dt" in metadata and not np.isclose(float(metadata["dt"]), dt_model, rtol=0, atol=1e-12):
        raise ValueError(f"Time step mismatch: dataset={metadata['dt']}, model={dt_model}")
    # Burgers stores the number of snapshots, whereas PureConvection stores
    # the number of one-step pairs per initial condition.
    if "n_steps_per_ic" in metadata:
        steps_per_ic = int(metadata["n_steps_per_ic"])
    elif "n_snaps" in metadata:
        steps_per_ic = int(metadata["n_snaps"]) - 1
    else:
        steps_per_ic = 1
    if steps_per_ic <= 0 or n % steps_per_ic != 0:
        raise ValueError("Invalid number of one-step pairs per initial condition")
    ic_types = metadata.get("ic_types")
    if ic_types is None or len(ic_types) != n:
        ic_types = ["not_recorded"] * n

    full_parts, noc_parts = [], []
    #predictions
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            batch = x[start:end].to(device=device)
            dt = torch.full((len(batch),), dt_model, device=device, dtype=batch.dtype)
            full = model(batch, dt)
            with without_coupling(model):
                no_c = model(batch, dt)
            full_parts.append(full.cpu().numpy())
            noc_parts.append(no_c.cpu().numpy())
            print(f"[one-step] {end}/{n}")

    pred_full = np.concatenate(full_parts, axis=0)
    pred_no_c = np.concatenate(noc_parts, axis=0)
    if not np.isfinite(pred_full).all() or not np.isfinite(pred_no_c).all():
        raise FloatingPointError("Non-finite predictions")

    rows_summary = [
        dataset_error_row("hybrid_full", pred_full, target),
        dataset_error_row("hybrid_no_c", pred_no_c, target),
    ]
    csv_summary = Path(save_test_metrics_csv(outdir, rows_summary))
    #errors evaluation
    errors_full = per_example_errors(pred_full, target)
    errors_noc = per_example_errors(pred_no_c, target)
    names = ("abs_l1", "rel_l1", "abs_linf", "rel_linf")

    example_rows = []
    for i in range(n):
        rec = {
            "example_index": i,
            "ic_index": i // steps_per_ic,
            "step_index": i % steps_per_ic,
            "ic_type": str(ic_types[i]),
        }
        for name in names:
            rec[name + "_full"] = float(errors_full[name][i])
            rec[name + "_no_c"] = float(errors_noc[name][i])
            rec["delta_" + name] = float(errors_noc[name][i] - errors_full[name][i])
        example_rows.append(rec)
    #csv writing
    csv_examples = outdir / "per_example_errors.csv"
    write_csv(csv_examples, list(example_rows[0].keys()), example_rows)

    group_rows = []
    for typ in sorted(set(map(str, ic_types))):
        indices = np.array([str(v) == typ for v in ic_types], dtype=bool)
        delta_l1 = errors_noc["rel_l1"][indices] - errors_full["rel_l1"][indices]
        delta_linf = errors_noc["rel_linf"][indices] - errors_full["rel_linf"][indices]
        group_rows.append({
            "ic_type": typ,
            "n_pairs": int(indices.sum()),
            "rel_l1_full_mean": float(errors_full["rel_l1"][indices].mean()),
            "rel_l1_no_c_mean": float(errors_noc["rel_l1"][indices].mean()),
            "delta_rel_l1_mean": float(delta_l1.mean()),
            "fraction_no_c_worse_l1": float((delta_l1 > 0).mean()),
            "rel_linf_full_mean": float(errors_full["rel_linf"][indices].mean()),
            "rel_linf_no_c_mean": float(errors_noc["rel_linf"][indices].mean()),
            "delta_rel_linf_mean": float(delta_linf.mean()),
        })
    csv_groups = outdir / "metrics_by_ic_type.csv"
    write_csv(csv_groups, list(group_rows[0].keys()), group_rows)

    print("\n[results] rel_l1_mean")
    for row in rows_summary:
        print(f"  {row['model']:15s} {row['rel_l1_mean']:.9g}")
    print("[delta] NO-C minus FULL (positive means ablation is worse)")
    print(f"  rel_L1 : {(errors_noc['rel_l1'] - errors_full['rel_l1']).mean():+.9g}")
    print(f"  rel_Linf: {(errors_noc['rel_linf'] - errors_full['rel_linf']).mean():+.9g}")
    print(f"\n[saved] {csv_summary}")
    print(f"[saved] {csv_examples}")
    print(f"[saved] {csv_groups}")


if __name__ == "__main__":
    main()

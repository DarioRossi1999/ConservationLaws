"""Compare total variation and maximum-principle violations in saved Burgers rollouts.

No model or numerical solver is rerun. Only new result files are written.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "results/Burgers/rollout/T4_seed2/trajectories_and_metrics.npz"
DEFAULT_OUTPUT = SCRIPT_DIR / "results/Burgers/total_variation/T4_seed2"


def total_variation(values):
    """Compute periodic spatial total variation for each condition and time."""
    return np.abs(np.roll(values, -1, axis=-1) - values).sum(axis=-1)


def maximum_principle_violation(values, initial_values):
    """Measure how far a prediction leaves the range of its initial condition."""
    initial_min = initial_values.min(axis=1)[:, None]
    initial_max = initial_values.max(axis=1)[:, None]

    above = np.maximum(values.max(axis=-1) - initial_max, 0.0)
    below = np.maximum(initial_min - values.min(axis=-1), 0.0)
    return np.maximum(above, below)


def save_csv(path, rows):
    """Save a list of records, with consistent column headers."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Burgers total-variation diagnostics")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(f"Trajectory file not found: {args.data}")

    with np.load(args.data, allow_pickle=False) as archive:
        times = archive["times"]
        initial_values = archive["u0_fine"]
        predictions = {
            "reference": archive["ref"],
            "full": archive["full"],
            "no_c": archive["no_c"],
        }

    reference = predictions["reference"]
    if any(values.shape != reference.shape for values in predictions.values()):
        raise ValueError("Reference, FULL and NO-C must have matching shapes.")
    if reference.shape[1] != len(times) or reference.shape[0] != len(initial_values):
        raise ValueError("Trajectory dimensions do not match times or initial conditions.")

    args.outdir.mkdir(parents=True, exist_ok=True)

    variations = {name: total_variation(values) for name, values in predictions.items()}
    violations = {
        name: maximum_principle_violation(values, initial_values)
        for name, values in predictions.items()
    }

    # Average over initial conditions at each recorded time.
    time_rows = []
    for index, time in enumerate(times):
        time_rows.append({
            "time": float(time),
            "tv_reference": float(variations["reference"][:, index].mean()),
            "tv_full": float(variations["full"][:, index].mean()),
            "tv_no_c": float(variations["no_c"][:, index].mean()),
            "mean_abs_tv_error_full": float(np.abs(
                variations["full"][:, index] - variations["reference"][:, index]
            ).mean()),
            "mean_abs_tv_error_no_c": float(np.abs(
                variations["no_c"][:, index] - variations["reference"][:, index]
            ).mean()),
            "mean_violation_full": float(violations["full"][:, index].mean()),
            "mean_violation_no_c": float(violations["no_c"][:, index].mean()),
        })
    save_csv(args.outdir / "tv_curves.csv", time_rows)

    summary_rows = []
    for name in ("reference", "full", "no_c"):
        tv = variations[name]
        violation = violations[name]
        tv_error = np.abs(tv - variations["reference"])
        summary_rows.append({
            "model": name,
            "n_conditions": len(initial_values),
            "n_times": len(times),
            "tv_initial_mean": float(tv[:, 0].mean()),
            "tv_final_mean": float(tv[:, -1].mean()),
            "mean_abs_tv_error": float(tv_error.mean()),
            "maximum_range_violation": float(violation.max()),
            "conditions_with_violation_above_1e-3": int((
                violation.max(axis=1) > 1e-3
            ).sum()),
        })
    save_csv(args.outdir / "tv_summary.csv", summary_rows)

    # Keep per-condition values so the summary can be independently checked.
    condition_rows = []
    for index in range(len(initial_values)):
        condition_rows.append({
            "ic": index + 1,
            "tv_error_full_time_mean": float(np.abs(
                variations["full"][index] - variations["reference"][index]
            ).mean()),
            "tv_error_no_c_time_mean": float(np.abs(
                variations["no_c"][index] - variations["reference"][index]
            ).mean()),
            "maximum_violation_full": float(violations["full"][index].max()),
            "maximum_violation_no_c": float(violations["no_c"][index].max()),
        })
    save_csv(args.outdir / "tv_per_condition.csv", condition_rows)

    for name in ("reference", "full", "no_c"):
        plt.plot(times, variations[name].mean(axis=0), label=name)
    plt.xlabel("Time")
    plt.ylabel("Mean periodic total variation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.outdir / "tv_curves.png", dpi=170)
    plt.close()

    for name in ("full", "no_c"):
        error = np.abs(variations[name] - variations["reference"])
        plt.plot(times, error.mean(axis=0), label=name)
    plt.xlabel("Time")
    plt.ylabel("Mean absolute TV difference from WENO-Z")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.outdir / "tv_error_curves.png", dpi=170)
    plt.close()

    print(f"Initial conditions: {len(initial_values)} | Recorded times: {len(times)}")
    for row in summary_rows:
        print(f"{row['model']:>9}: final TV={row['tv_final_mean']:.6f} | "
              f"mean |TV - TV_ref|={row['mean_abs_tv_error']:.6f} | "
              f"max range violation={row['maximum_range_violation']:.6f}")
    print(f"[saved] {args.outdir.resolve()}")


if __name__ == "__main__":
    main()

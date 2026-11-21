"""Summarize all four LGNO mixer contributions and paired near/far differences.

Reads the CSV produced by task3_analyze_coupling.py. Does not load models,
run simulations, or change any original repository files.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BRANCHES = {
    "hidden": "M_h_h_l2",
    "global": "M_g_g_l2",
    "local": "M_local_l_l2",
    "coupling": "M_c_c_l2",
}
PAIRS = (("jump_near", "jump_far"), ("shock_near", "shock_far"))


def write_csv(path, rows):
    """Write a CSV with stable column ordering."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def read_measurements(path):
    """Read saved branch norms and calculate descriptive shares."""
    with path.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    if not source_rows:
        raise ValueError("No measurements found")

    # 1. Read the four mixer contributions for each IC, time, layer and region.
    measurements = []
    for row in source_rows:
        magnitudes = {name: float(row[field]) for name, field in BRANCHES.items()}
        # Shares describe relative magnitudes; they are not additive output effects.
        total = sum(magnitudes.values())
        measurements.append({
            "condition": int(row["condition"]),
            "time": float(row["time"]),
            "layer": int(row["layer"]),
            "region": row["region"],
            "norms": magnitudes,
            "shares": {name: value / total if total else np.nan
                       for name, value in magnitudes.items()},
        })
    return measurements

def summarize_by_condition(measurements):
    """Average over times first, then weight each condition equally."""
    # 2. Average the different times of the same initial condition first.
    groups = defaultdict(list)
    for measurement in measurements:
        key = (measurement["layer"], measurement["region"], measurement["condition"])
        groups[key].append(measurement)

    condition_means = {}
    for key, group in groups.items():
        condition_means[key] = {
            "norms": {name: np.mean([item["norms"][name] for item in group])
                      for name in BRANCHES},
            "shares": {name: np.mean([item["shares"][name] for item in group])
                       for name in BRANCHES},
        }

    # Then give each independent initial condition the same weight.
    region_groups = defaultdict(list)
    for (layer, region, condition), values in condition_means.items():
        region_groups[(layer, region)].append(values)

    summaries = []
    for (layer, region), cases in sorted(region_groups.items()):
        for branch in BRANCHES:
            summaries.append({
                "layer": layer,
                "region": region,
                "branch": branch,
                "n_conditions": len(cases),
                "mean_norm": float(np.mean([case["norms"][branch] for case in cases])),
                "mean_share": float(np.mean([case["shares"][branch] for case in cases])),
            })
    return summaries


def confidence_interval(values, rng):
    """Bootstrap whole independent conditions, not time snapshots."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan, np.nan
    # 2000 bootstrap draws: resample ICs, not correlated time snapshots.
    indices = rng.integers(0, len(values), size=(2000, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def paired_comparisons(measurements):
    """Pair near/far measurements at identical condition, time and layer."""
    # 3. Match near and far measurements from the same IC, time and layer.
    by_snapshot = defaultdict(dict)
    for item in measurements:
        key = (item["condition"], item["time"], item["layer"])
        by_snapshot[key][item["region"]] = item

    rng = np.random.default_rng(20261108)
    results = []
    for near_name, far_name in PAIRS:
        pairs = defaultdict(list)
        for (condition, time, layer), regions in by_snapshot.items():
            if near_name in regions and far_name in regions:
                pairs[(layer, condition)].append((regions[near_name], regions[far_name]))
        for layer in sorted(set(layer for layer, _ in pairs)):
            conditions = sorted(condition for level, condition in pairs if level == layer)
            for branch in BRANCHES:
                near_norms, far_norms, near_shares, far_shares = [], [], [], []
                for condition in conditions:
                    observations = pairs[(layer, condition)]
                    near_norms.append(np.mean([a["norms"][branch] for a, _ in observations]))
                    far_norms.append(np.mean([b["norms"][branch] for _, b in observations]))
                    near_shares.append(np.mean([a["shares"][branch] for a, _ in observations]))
                    far_shares.append(np.mean([b["shares"][branch] for _, b in observations]))

                # Compare the IC means, then bootstrap the paired differences.
                norm_differences = [a - b for a, b in zip(near_norms, far_norms)]
                share_differences = [a - b for a, b in zip(near_shares, far_shares)]
                near_mean, far_mean = np.mean(near_norms), np.mean(far_norms)
                norm_ci = confidence_interval(norm_differences, rng)
                share_ci = confidence_interval(share_differences, rng)
                results.append({
                    "layer": layer,
                    "branch": branch,
                    "near_region": near_name,
                    "far_region": far_name,
                    "n_conditions": len(conditions),
                    "mean_norm_near": float(near_mean),
                    "mean_norm_far": float(far_mean),
                    "norm_near_over_far": float(near_mean / far_mean) if far_mean > 0 else np.nan,
                    "paired_norm_near_minus_far": float(np.mean(norm_differences)),
                    "norm_diff_ci_low": norm_ci[0],
                    "norm_diff_ci_high": norm_ci[1],
                    "mean_share_near": float(np.mean(near_shares)),
                    "mean_share_far": float(np.mean(far_shares)),
                    "paired_share_near_minus_far": float(np.mean(share_differences)),
                    "share_diff_ci_low": share_ci[0],
                    "share_diff_ci_high": share_ci[1],
                })
    return results


def make_plots(summary, paired, output):
    """Plot norms for every branch and the near/far contrast by layer."""
    # 4. Plot the layerwise branch magnitudes and near/far ratios.
    regions = sorted({row["region"] for row in summary})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, branch in zip(axes.flat, BRANCHES):
        for region in regions:
            group = [row for row in summary if row["branch"] == branch and row["region"] == region]
            ax.plot([row["layer"] for row in group], [row["mean_norm"] for row in group],
                    marker="o", label=region)
        ax.set_title(branch)
        ax.set_xticks([1, 2, 3, 4])
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Mean channel-L2 norm")
    axes[1, 0].set_ylabel("Mean channel-L2 norm")
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "all_branch_norms.png", dpi=160)
    plt.close(fig)

    if paired:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for branch in BRANCHES:
            group = [row for row in paired if row["branch"] == branch]
            ax.plot([row["layer"] for row in group], [row["norm_near_over_far"] for row in group],
                    marker="o", label=branch)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Near / far mean branch norm")
        ax.set_xticks([1, 2, 3, 4])
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "branch_near_far_ratios.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Saved contributions_per_case.csv")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Missing input: {args.input}")
    # 5. Compute summaries from saved CSVs
    measurements = read_measurements(args.input)
    summary = summarize_by_condition(measurements)
    paired = paired_comparisons(measurements)
    # Save the same CSV columns and figures.
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "all_branch_summary.csv", summary)
    write_csv(args.outdir / "paired_near_far.csv", paired)
    make_plots(summary, paired, args.outdir)
    print(f"[saved] {args.outdir}")
    print("[summary] All branches, all layers:")
    for row in summary:
        print(f"  layer={row['layer']} {row['region']:17s} {row['branch']:8s} "
              f"norm={row['mean_norm']:.5g} share={row['mean_share']:.3f} "
              f"n_ic={row['n_conditions']}")
    if paired:
        print("[paired] Near / far norm ratios (same condition and time):")
        for row in paired:
            print(f"  layer={row['layer']} {row['branch']:8s} "
                  f"near/far={row['norm_near_over_far']:.3f} "
                  f"share_diff={row['paired_share_near_minus_far']:.3f}")
    print("Shares are descriptive ratios of branch magnitudes, not additive output contributions.")


if __name__ == "__main__":
    main()

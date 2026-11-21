"""Compare LGNO prediction errors near and far from detected Burgers fronts.

Read saved trajectories and WENO-Z front positions. No model is rerun.
The original repository, model checkpoints and datasets are not modified.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "results/Burgers/rollout/T4_seed2/trajectories_and_metrics.npz"
DEFAULT_FRONTS = SCRIPT_DIR / "results/Burgers/shock_detection_weno"
DEFAULT_OUTPUT = SCRIPT_DIR / "results/Burgers/shock_error_analysis"


def save_csv(path, rows):
    """Write a list of dictionaries to a CSV file."""
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_fronts(path):
    """Read detected positions and first-shock times for each initial condition."""
    positions = defaultdict(list)
    first_shock_times = {}

    # Front CSVs use IC numbers starting at 1; Python arrays start at 0.
    # Round times so that CSV and saved trajectory timestamps use the same keys.
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            condition = int(row["ic"]) - 1
            time = round(float(row["time"]), 8)
            positions[(condition, time)].append(float(row["x_front_weno"]))
            first_shock_times[condition] = float(row["t_star"])

    return positions, first_shock_times


def compare_regions(reference, full, no_c, grid, times, positions,
                    first_shock_times, radius_cells, file_name):
    """Calculate errors for each condition and time, then average by condition."""
    n_conditions, n_times, n_points = reference.shape
    # 1. Convert the chosen near-region radius from grid cells to physical distance.
    dx = float(grid[1] - grid[0])
    domain_length = n_points * dx
    radius = radius_cells * dx

    detailed_rows = []
    averages_by_condition = []
    coverage = []
    missing_fronts = 0

    # 2. Examine each independent initial condition separately.
    for condition in range(n_conditions):
        errors_for_condition = []
        shock_time = first_shock_times.get(condition)

        if shock_time is None:
            continue

        # Use only observations strictly after the first predicted shock time.
        for time_index in range(1, n_times):
            time = float(times[time_index])
            if time <= shock_time:
                continue

            # Skip times without detected fronts; they have no near-region mask.
            fronts = positions.get((condition, round(time, 8)), [])
            if not fronts:
                missing_fronts += 1
                continue

            # Periodic distance from every grid point to the closest front.
            distances = np.abs(grid[:, None] - np.array(fronts)[None, :])
            distances = np.minimum(distances, domain_length - distances)
            near = np.min(distances, axis=1) <= radius + 1e-12

            # Both regions must contain at least one grid point.
            if not near.any() or near.all():
                continue

            # 3. Calculate the absolute errors on the same WENO-Z reference state.
            # Near and far are complementary spatial regions for this profile.
            error_full = np.abs(full[condition, time_index] - reference[condition, time_index])
            error_no_c = np.abs(no_c[condition, time_index] - reference[condition, time_index])

            # Average over spatial cells first, separately for each region/model.
            measurements = np.array([
                error_full[near].mean(),
                error_no_c[near].mean(),
                error_full[~near].mean(),
                error_no_c[~near].mean(),
            ])
            errors_for_condition.append(measurements)
            coverage.append(float(near.mean()))

            detailed_rows.append({
                "front_file": file_name,
                "radius_dx": radius_cells,
                "ic": condition + 1,
                "time": f"{time:.8f}",
                "n_fronts": len(fronts),
                "n_near_cells": int(near.sum()),
                "coverage_fraction": float(near.mean()),
                "full_near_abs_l1": measurements[0],
                "no_c_near_abs_l1": measurements[1],
                "delta_near": measurements[1] - measurements[0],
                "full_far_abs_l1": measurements[2],
                "no_c_far_abs_l1": measurements[3],
                "delta_far": measurements[3] - measurements[2],
            })

        if errors_for_condition:
            # 4. Average over time within each IC before averaging across ICs.
            # This avoids treating correlated times as independent observations.
            averages_by_condition.append(np.mean(errors_for_condition, axis=0))

    if not averages_by_condition:
        raise ValueError(f"No valid front regions for {file_name}, radius {radius_cells} dx")

    # 5. Average across independent ICs, then compare NO-C with FULL.
    condition_means = np.array(averages_by_condition)
    overall = condition_means.mean(axis=0)
    delta_near = float(overall[1] - overall[0])
    delta_far = float(overall[3] - overall[2])

    # 6. Estimate uncertainty in (NO-C - FULL)near - (NO-C - FULL)far.
    # Bootstrap complete ICs, preserving their paired near/far measurements.
    condition_difference = (
        (condition_means[:, 1] - condition_means[:, 0])
        - (condition_means[:, 3] - condition_means[:, 2])
    )
    if len(condition_means) < 2:
        # A single initial condition cannot provide a meaningful bootstrap interval.
        confidence_low = confidence_high = np.nan
    else:
        rng = np.random.default_rng(12345)
        samples = rng.integers(0, len(condition_means), size=(2000, len(condition_means)))
        bootstrap_means = condition_difference[samples].mean(axis=1)
        confidence_low, confidence_high = np.quantile(bootstrap_means, [0.025, 0.975])

    # The bootstrap interval is for the difference between regions, not the ratio.
    summary = {
        "front_file": file_name,
        "radius_dx": radius_cells,
        "n_ic": len(condition_means),
        "n_ic_time_pairs": len(detailed_rows),
        "post_shock_pairs_without_front": missing_fronts,
        "mean_coverage": float(np.mean(coverage)),
        "full_near_abs_l1": overall[0],
        "no_c_near_abs_l1": overall[1],
        "delta_near": delta_near,
        "full_far_abs_l1": overall[2],
        "no_c_far_abs_l1": overall[3],
        "delta_far": delta_far,
        "delta_near_over_delta_far": delta_near / delta_far if delta_far else np.nan,
        "delta_near_minus_far": delta_near - delta_far,
        "bootstrap_95ci_low": confidence_low,
        "bootstrap_95ci_high": confidence_high,
        "fraction_ic_delta_near_positive": float(np.mean(
            condition_means[:, 1] > condition_means[:, 0]
        )),
    }
    return detailed_rows, summary


def main():
    parser = argparse.ArgumentParser(description="Compare prediction errors near detected Burgers fronts")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--front-dir", type=Path, default=DEFAULT_FRONTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--radii", type=int, nargs="+", default=[2, 4, 8],
                        help="Region radius expressed in grid spacings")
    args = parser.parse_args()

    # 1. Read paths and spatial radii from the command line.

    if not args.data.is_file():
        parser.error(f"Trajectory file not found: {args.data}")
    if any(radius < 1 for radius in args.radii):
        parser.error("All radii must be positive integers")

    # 2. Include each saved front-detection threshold configuration.
    front_files = sorted(args.front_dir.glob("weno_fronts_all_thr*_peak*.csv"))
    if not front_files:
        parser.error("No front CSVs found. Run detect_weno_shocks.py --all first.")

    # 3. Load previously computed WENO-Z, FULL and NO-C trajectories.
    with np.load(args.data, allow_pickle=False) as archive:
        grid = archive["x_low"]
        times = archive["times"]
        reference = archive["ref"]
        full = archive["full"]
        no_c = archive["no_c"]

    if reference.shape != full.shape or reference.shape != no_c.shape:
        raise ValueError("The three sets of trajectories must have the same shape")
    if reference.shape != (len(reference), len(times), len(grid)):
        raise ValueError("Trajectory shape does not match the spatial and temporal grids")

    detailed_rows = []
    summary_rows = []

    # 4. Compare near and far regions for every threshold and requested radius.
    for front_file in front_files:
        positions, first_shock_times = load_fronts(front_file)

        for radius in sorted(set(args.radii)):
            details, summary = compare_regions(
                reference, full, no_c, grid, times, positions,
                first_shock_times, radius, front_file.name,
            )
            detailed_rows.extend(details)
            summary_rows.append(summary)

            print(f"{front_file.name} | radius={radius} dx | conditions={summary['n_ic']}")
            print(f"  Extra error near fronts: {summary['delta_near']:.8g}")
            print(f"  Extra error far from fronts: {summary['delta_far']:.8g}")
            print(f"  Near/far ratio: {summary['delta_near_over_delta_far']:.2f}x")
            if summary["n_ic"] < 2:
                print("  Bootstrap interval unavailable with only one initial condition.")
            else:
                print(f"  95% bootstrap interval for near-minus-far: "
                      f"[{summary['bootstrap_95ci_low']:.8g}, "
                      f"{summary['bootstrap_95ci_high']:.8g}]")

    # 5. Save one detailed CSV and one compact summary CSV.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.out_dir / "shock_region_summary.csv", summary_rows)
    save_csv(args.out_dir / "per_ic_time_regions.csv", detailed_rows)
    print(f"[saved] {args.out_dir / 'shock_region_summary.csv'}")
    print(f"[saved] {args.out_dir / 'per_ic_time_regions.csv'}")
    print("Note: front locations are numerical WENO-Z estimates, not exact shocks.")


if __name__ == "__main__":
    main()

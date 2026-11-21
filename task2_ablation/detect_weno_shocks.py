"""Detect numerical shock candidates in saved Burgers WENO-Z trajectories.

The first shock time and position are estimated from the Fourier initial data.
At later times, descending fronts are located in the WENO-Z reference solution.
These front positions are numerical estimates, not exact shock locations.

Only CSV results are written. The original repository, datasets, and model
checkpoints are never modified.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


# 1. Find the saved Burgers trajectories and choose the default CSV folder.
# Relative paths keep the script usable from any working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "results/Burgers/rollout/T4_seed2/trajectories_and_metrics.npz"
OUTPUT_DIR = SCRIPT_DIR / "results/Burgers/shock_detection_weno"

# Keep these column names unchanged for compatibility with later analyses.
CSV_COLUMNS = [
    "ic", "time", "t_star", "x_star", "fourier_kmax", "shock_id",
    "x_front_weno", "cell_drop", "cell_drop_over_range", "threshold_cell_drop",
]


def first_shock(initial_values):
    """Estimate the first shock time and position using characteristics."""
    n = len(initial_values)
    coefficients = np.fft.rfft(initial_values) / n

    # Reconstruct the smooth initial condition using its non-negligible
    # Fourier modes. The original Burgers generator uses only a few modes.
    significant_modes = np.flatnonzero(
        np.abs(coefficients) > 1e-11 * np.max(np.abs(coefficients))
    )
    max_mode = int(significant_modes[-1])

    if max_mode >= n // 2:
        raise ValueError("The initial condition contains significant high Fourier modes.")

    # Differentiate the Fourier series and sample it on a finer grid
    # to find where the initial slope becomes most negative.
    fine_n = 32 * n
    modes = np.arange(max_mode + 1)
    derivative_coefficients = np.zeros(fine_n // 2 + 1, dtype=complex)
    derivative_coefficients[:max_mode + 1] = (
        fine_n * (2j * np.pi * modes) * coefficients[:max_mode + 1]
    )
    derivative = np.fft.irfft(derivative_coefficients, n=fine_n)

    # Locate the minimum slope, then refine its position using the two
    # neighboring grid values and a simple parabolic interpolation.
    min_index = int(np.argmin(derivative))
    left = derivative[(min_index - 1) % fine_n]
    center = derivative[min_index]
    right = derivative[(min_index + 1) % fine_n]
    curvature = left - 2 * center + right

    shift = 0.0
    if curvature > 0:
        shift = 0.5 * (left - right) / curvature
    shift = float(np.clip(shift, -0.5, 0.5))
    initial_position = ((min_index + shift) / fine_n) % 1.0

    # Evaluate both u0 and its derivative at this refined position.
    modes = np.arange(1, max_mode + 1)
    terms = coefficients[1:max_mode + 1] * np.exp(
        2j * np.pi * modes * initial_position
    )
    initial_value = float(coefficients[0].real + 2 * np.real(np.sum(terms)))
    min_slope = float(2 * np.real(np.sum(2j * np.pi * modes * terms)))

    if min_slope >= 0:
        return np.inf, np.nan, max_mode

    # Burgers characteristics: t* = -1/min(u0'), x* = xi + t* u0(xi).
    shock_time = -1.0 / min_slope
    shock_position = (initial_position + shock_time * initial_value) % 1.0
    return float(shock_time), float(shock_position), max_mode


# A piecewise-constant initial profile may already contain a shock at t=0.
def first_piecewise_shock(initial_values):
    """Locate initial descending jumps in a periodic piecewise-constant profile."""
    n = len(initial_values)
    drops = initial_values - np.roll(initial_values, -1)
    amplitude = float(np.ptp(initial_values))

    # Keep descending jumps of at least 10% of the profile amplitude.
    # Use the strongest one to report a representative initial position.
    if amplitude == 0:
        return np.inf, np.nan, ""
    strong_jumps = np.flatnonzero(drops >= 0.1 * amplitude)
    if not len(strong_jumps):
        return np.inf, np.nan, ""

    # The first point of the right state approximates the discontinuity location.
    largest_jump = int(strong_jumps[np.argmax(drops[strong_jumps])])
    shock_position = ((largest_jump + 1) / n) % 1.0
    return 0.0, float(shock_position), ""


def find_fronts(values, grid, threshold_fraction, max_peak_fraction, min_separation):
    """Find strong local maxima of descending cell-to-cell drops."""
    n = len(values)
    dx = float(grid[1] - grid[0])

    # Compare each cell with its next neighbor, including the periodic wrap.
    # A positive value means the solution decreases across that cell edge.
    drops = values - np.roll(values, -1)
    amplitude = float(np.ptp(values))
    max_drop = max(float(np.max(drops)), 0.0)

    # A candidate must exceed both the amplitude-based requirement and
    # the peak-relative requirement. The larger threshold is used.
    threshold = max(
        threshold_fraction * amplitude,
        max_peak_fraction * max_drop,
    )
    strong = (drops > 0) & (drops >= threshold)
    local_maxima = (
        (drops >= np.roll(drops, 1))
        & (drops >= np.roll(drops, -1))
    )
    candidates = np.flatnonzero(strong & local_maxima)

    # Select the strongest peaks first. Suppress weaker candidates closer
    # than min_separation cells, using periodic distances.
    selected = []
    for index in sorted(candidates, key=lambda i: -drops[i]):
        distances = [
            min(abs(index - other), n - abs(index - other))
            for other in selected
        ]
        if all(distance >= min_separation for distance in distances):
            selected.append(int(index))

    # Estimate each front position at the midpoint of the cell edge.
    # Sort positions to give a consistent spatial order in the CSV.
    fronts = []
    for index in selected:
        position = (float(grid[index]) + dx / 2) % 1.0
        relative_drop = float(drops[index] / amplitude) if amplitude > 0 else 0.0
        fronts.append((position, float(drops[index]), relative_drop))

    return sorted(fronts), float(threshold)


def main():
    parser = argparse.ArgumentParser(
        description="Detect numerical shock candidates in Burgers WENO-Z trajectories"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--ic", type=int, default=1, help="Initial condition, starting from 1")
    parser.add_argument("--times", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--all", action="store_true", help="Process all conditions and times")
    parser.add_argument("--threshold-fraction", type=float, default=0.06)
    parser.add_argument("--max-peak-fraction", type=float, default=0.12)
    parser.add_argument("--min-separation", type=int, default=4)
    parser.add_argument("--ic-type", choices=("fourier", "piecewise_constant"), default="fourier")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(f"Data file not found: {args.data}")
    if not (0 < args.threshold_fraction < 1 and 0 < args.max_peak_fraction < 1):
        parser.error("Both threshold fractions must be between 0 and 1.")
    if args.min_separation < 1:
        parser.error("Minimum separation must be at least one cell.")

    # 2. Load previously saved WENO-Z trajectories and their initial states.
    # No LGNO inference or new WENO-Z simulation is performed here.
    with np.load(args.data, allow_pickle=False) as archive:
        grid = archive["x_low"]
        times = archive["times"]
        reference = archive["ref"]
        initial_conditions = archive["u0_fine"]
        initial_types = archive["ic_types"] if "ic_types" in archive else None

    if initial_types is not None:
        labels = set(map(str, initial_types))
        if args.ic_type == "fourier" and "piecewise_constant" in labels:
            parser.error("The file contains piecewise-constant data: set --ic-type piecewise_constant.")
        if args.ic_type == "piecewise_constant" and "fourier_random" in labels:
            parser.error("The file contains Fourier data: use the default --ic-type fourier.")

    if reference.shape != (len(initial_conditions), len(times), len(grid)):
        raise ValueError("Unexpected array dimensions in the NPZ file.")

    # 3. Select either all conditions and times, or one condition at
    # the explicitly requested times. Smooth profiles skip t=0.
    if args.all:
        condition_indices = range(len(initial_conditions))
        time_indices = range(
            0 if args.ic_type == "piecewise_constant" else 1,
            len(times),
        )  # Exclude t = 0.
    else:
        if not 1 <= args.ic <= len(initial_conditions):
            parser.error(f"--ic must be between 1 and {len(initial_conditions)}.")
        condition_indices = [args.ic - 1]
        time_indices = []
        for requested_time in args.times:
            index = int(np.argmin(np.abs(times - requested_time)))
            if not np.isclose(times[index], requested_time, atol=1e-10):
                parser.error(f"Time {requested_time} is not available in the data.")
            if (index > 0 or args.ic_type == "piecewise_constant") and index not in time_indices:
                time_indices.append(index)

    # 4. Analyze each initial condition and calculate its first-shock data.
    # These values will be repeated in the CSV for every detected front.
    rows = []
    empty_after_first_shock = 0

    for condition_index in condition_indices:
        if args.ic_type == "piecewise_constant":
            shock_time, shock_position, max_mode = first_piecewise_shock(
                initial_conditions[condition_index]
            )
        else:
            shock_time, shock_position, max_mode = first_shock(
                initial_conditions[condition_index]
            )
        if not args.all:
            if args.ic_type == "piecewise_constant":
                print(
                    f"IC {condition_index + 1}: initial descending shock "
                    f"t*={shock_time:.8f}, x*={shock_position:.8f}"
                )
            else:
                print(
                    f"IC {condition_index + 1}: first shock t*={shock_time:.8f}, "
                    f"x*={shock_position:.8f} (Fourier K={max_mode})"
                )

        for time_index in time_indices:
            time = float(times[time_index])
            fronts = []
            threshold = np.nan

            # 5. Look for descending fronts only after shock formation.
            # Piecewise-constant profiles may already have a front at t=0.
            if time > shock_time or (args.ic_type == "piecewise_constant" and time == 0 and shock_time == 0):
                fronts, threshold = find_fronts(
                    reference[condition_index, time_index],
                    grid,
                    args.threshold_fraction,
                    args.max_peak_fraction,
                    args.min_separation,
                )
                if not fronts:
                    empty_after_first_shock += 1

            if not args.all:
                positions = [round(position, 6) for position, _, _ in fronts]
                print(f"  t={time:.2f}: {len(fronts)} numerical fronts, x={positions}")

            # 6. Write one record per detected front. shock_id indicates
            # spatial order at this time, not tracking across different times.
            for front_number, (position, drop, relative_drop) in enumerate(fronts, 1):
                rows.append({
                    "ic": condition_index + 1,
                    "time": f"{time:.8f}",
                    "t_star": f"{shock_time:.10f}",
                    "x_star": f"{shock_position:.10f}",
                    "fourier_kmax": max_mode,
                    "shock_id": front_number,
                    "x_front_weno": f"{position:.10f}",
                    "cell_drop": f"{drop:.9g}",
                    "cell_drop_over_range": f"{relative_drop:.6g}",
                    "threshold_cell_drop": f"{threshold:.9g}",
                })

    # 7. Save the front positions in the CSV format expected by
    # compare_shock_regions.py. The file name records the detection thresholds.
    output_dir = args.out_dir
    if output_dir is None:
        output_dir = OUTPUT_DIR / "piecewise_constant" if args.ic_type == "piecewise_constant" else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = (
        f"thr{args.threshold_fraction:.3f}_peak{args.max_peak_fraction:.3f}"
        .replace(".", "p")
    )
    selection = "all" if args.all else f"ic{args.ic:02d}"
    output_file = output_dir / f"weno_fronts_{selection}_{parameters}.csv"

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[saved] {output_file}")
    print(
        f"Total candidates: {len(rows)}; "
        f"time samples after first shock without detected fronts: "
        f"{empty_after_first_shock}."
    )
    print("Note: WENO-Z front positions are numerical estimates, not exact shock locations.")


if __name__ == "__main__":
    main()

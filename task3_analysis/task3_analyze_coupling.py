"""Inspect original LGNO matrices and their actual contributions to hidden-state updates.

This is a read-only analysis of pretrained checkpoints and saved reference data.
No training, ablation, or changes to the original repository are performed.

Run from the repository root:
    python .\task3_analysis\task3_analyze_coupling.py --problem PureConvection
    python .\task3_analysis\task3_analyze_coupling.py --problem Burgers
    python .\task3_analysis\task3_analyze_coupling.py --problem Burgers --source step
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Load the original pretrained LGNO model without editing its source
from conslaw.checkpoints import load_hybrid_step_map_1d

# The original evaluation scripts use double precision.
torch.set_default_dtype(torch.float64)

# Input paths: original checkpoints and reference states from Tasks 1 and 2
# Results will be written only to the selected Task 3 output directory
CHECKPOINTS = {
    "PureConvection": ROOT / "PureConvection/checkpoints/pureconvection_hybrid_flowmap_dt.pt",
    "Burgers": ROOT / "Burgers/checkpoints/burgers_hybrid_flowmap_dt.pt",
}
PURE_DATA = ROOT / "PureConvection/dataset/test_pv.pt"
BURGERS_DATA = ROOT / "task2_ablation/results/Burgers/rollout/T4_seed2/trajectories_and_metrics.npz"
BURGERS_FRONTS = ROOT / "task2_ablation/results/Burgers/shock_detection_weno/weno_fronts_all_thr0p060_peak0p120.csv"
STEP_DATA = ROOT / "task2_ablation/results/Burgers/piecewise_constant/step_1_to_0_T0p5/trajectories_and_metrics.npz"
STEP_FRONTS = ROOT / "task2_ablation/results/Burgers/shock_detection_weno/piecewise_constant/weno_fronts_all_thr0p060_peak0p120.csv"

BRANCHES = ("h", "g", "local", "coupling")


def save_csv(path: Path, rows: list[dict]) -> None:
    """Write records to CSV while keeping all columns in their original order."""
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path: Path) -> str:
    """Compute a SHA-256 fingerprint without changing the file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_statistics(problem: str, backbone) -> tuple[list[dict], list[dict]]:
    """Measure each mixer block (Mh,Mg,Ml,Mc) and the two learned coupling transforms (Ag,Al) for every layer."""
    rows = []
    singular_rows = []
    width = backbone.width

    # 1. Inspect fixed weights, independently of any input profile.
    # Each mixer maps [h, g, local, coupling] to an updated hidden state.
    for layer in range(backbone.n_layers):
        mixer = backbone.mix[layer]
        mixer_matrix = mixer.weight.detach().cpu().numpy()[:, :, 0]
        transforms = [
            ("M_h", mixer_matrix[:, :width], None),
            ("M_g", mixer_matrix[:, width:2 * width], None),
            ("M_local", mixer_matrix[:, 2 * width:3 * width], None),
            ("M_c", mixer_matrix[:, 3 * width:4 * width], None),
            ("A_g", backbone.w_g[layer].weight.detach().cpu().numpy()[:, :, 0], backbone.w_g[layer].bias),
            ("A_l", backbone.w_l[layer].weight.detach().cpu().numpy()[:, :, 0], backbone.w_l[layer].bias),
        ]
        # Measure the Frobenius and spectral norms of each weight block.
        # Keep singular values for the existing CSV output; no rank metric is needed.
        # These norms describe fixed weights, not their actual response to data.
        for name, matrix, bias in transforms:
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            rows.append({
                "problem": problem,
                "layer": layer + 1,
                "component": name,
                "frobenius_norm": float(np.linalg.norm(matrix)),
                "spectral_norm": float(singular_values[0]),
                "bias_l2": float(torch.linalg.vector_norm(bias).item()) if bias is not None else "",
            })
            for index, value in enumerate(singular_values, start=1):
                singular_rows.append({
                    "problem": problem,
                    "layer": layer + 1,
                    "component": name,
                    "singular_index": index,
                    "singular_value": float(value),
                })
        rows.append({
            "problem": problem,
            "layer": layer + 1,
            "component": "M_bias",
            "frobenius_norm": "",
            "spectral_norm": "",
            "bias_l2": float(torch.linalg.vector_norm(mixer.bias).item()),
        })
    return rows, singular_rows

# 2. Read Burgers front locations detected earlier from WENO-Z snapshots.
# The CSV numbers conditions from 1; our arrays index them from 0.
def load_front_positions(path: Path) -> dict[tuple[int, float], list[float]]:
    """Read the previously detected WENO-Z fronts, indexed by zero-based condition."""
    positions = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["ic"]) - 1, round(float(row["time"]), 8))
            positions[key].append(float(row["x_front_weno"]))
    return positions


def periodic_near_mask(grid: np.ndarray, fronts: list[float], radius_cells: int) -> np.ndarray:
    """Select grid points within the chosen periodic distance from any front."""
    if not fronts:
        raise ValueError("No fronts available to define a near-front region")
    dx = float(grid[1] - grid[0])
    length = dx * len(grid)
    # Respect periodic boundaries: x near 0 can also be near a front at x near 1.
    distances = np.abs(grid[:, None] - np.asarray(fronts)[None, :])
    distances = np.minimum(distances, length - distances)
    return np.min(distances, axis=1) <= radius_cells * dx + 1e-12


def discontinuity_positions(values: np.ndarray, grid: np.ndarray) -> list[float]:
    """Locate large positive or negative jumps in a periodic advection profile."""
    # Advection preserves both upward and downward jumps, so use absolute jumps.
    # Keep sufficiently large local maxima and avoid counting close peaks twice.
    changes = np.abs(values - np.roll(values, -1))
    amplitude = float(np.ptp(values))
    threshold = max(0.06 * amplitude, 0.12 * float(np.max(changes)))
    local_maxima = (changes >= np.roll(changes, 1)) & (changes >= np.roll(changes, -1))
    candidates = np.flatnonzero((changes > 0) & (changes >= threshold) & local_maxima)
    selected = []
    n = len(values)
    for index in sorted(candidates.tolist(), key=lambda j: -changes[j]):
        if all(min(abs(index - old), n - abs(index - old)) >= 4 for old in selected):
            selected.append(index)
    dx = float(grid[1] - grid[0])
    return [float((grid[index] + dx / 2) % (dx * n)) for index in selected]

# 3. At each spatial point, take the L2 norm across hidden channels.
# Then average those norms over the chosen near/far/smooth region.
def mean_channel_l2(tensor: torch.Tensor, mask: torch.Tensor) -> float:
    """Average the Euclidean channel norm over selected spatial positions."""
    norms = torch.linalg.vector_norm(tensor, dim=1)[0]
    return float(norms[mask].mean().item())

# 4. Inspect one WENO-Z state by running the original FULL model once.
# Hooks observe the mixer inputs and collect measurements for every layer.
def analyze_input(model, values: np.ndarray, regions: dict[str, np.ndarray],
                  problem: str, source: str, condition: int, time: float,
                  dt_model: float, output_rows: list[dict]) -> float:
    """Read actual mixer inputs with temporary hooks; leave the forward pass unchanged."""
    backbone = model.backbone
    width = backbone.width
    n = len(values)
    mask_tensors = {}
    # Translate the selected spatial regions into masks used by PyTorch.
    for name, mask in regions.items():
        if len(mask) != n or not mask.any():
            continue
        mask_tensors[name] = torch.as_tensor(mask, dtype=torch.bool)
    if not mask_tensors:
        raise ValueError("All requested spatial regions are empty")

    reconstruction_errors = []
    hooks = []

    def make_hook(layer):
        def inspect_mixer(mixer, inputs):
            z = inputs[0]
            if z.shape != (1, 4 * width, n):
                raise ValueError(f"Unexpected mixer input shape: {tuple(z.shape)}")
            # Split [h, g, local, c] and the mixer weights into four blocks.
            # Measure the actual vectors M_h h, M_g g, M_local l and M_c c.
            features = list(torch.chunk(z, 4, dim=1))
            weight_blocks = list(torch.chunk(mixer.weight, 4, dim=1))
            contributions = [F.conv1d(feature, block) for feature, block in zip(features, weight_blocks)]

            # Check that c really equals (A_g g + b_g) * (A_l l + b_l).
            # Also check that the four contributions plus bias reproduce the mixer.
            # These checks do not change the model or its predictions.
            transformed_g = backbone.w_g[layer](features[1])
            transformed_l = backbone.w_l[layer](features[2])
            coupling_error = torch.max(torch.abs(transformed_g * transformed_l - features[3])).item()
            mixer_update = torch.stack(contributions).sum(dim=0)
            if mixer.bias is not None:
                mixer_update = mixer_update + mixer.bias.view(1, -1, 1)
            direct_output = F.conv1d(z, mixer.weight, mixer.bias)
            update_error = torch.max(torch.abs(direct_output - mixer_update)).item()
            reconstruction_errors.extend([float(coupling_error), float(update_error)])
            if max(coupling_error, update_error) > 1e-8:
                raise RuntimeError("Mixer or coupling reconstruction failed")

            # Measure the branch vectors only on the grid cells in each region.
            # The same FULL state is used for near and far; only the mask changes.
            for region, mask in mask_tensors.items():
                feature_sizes = [mean_channel_l2(item, mask) for item in features]
                contribution_sizes = [mean_channel_l2(item, mask) for item in contributions]
                # This share is descriptive: it excludes mixer bias and residual +h,
                # and vector contributions can cancel one another.
                denominator = sum(contribution_sizes)
                row = {
                    "problem": problem,
                    "source": source,
                    "condition": condition,
                    "time": float(time),
                    "layer": layer + 1,
                    "region": region,
                    "n_grid_points": int(mask.sum().item()),
                    "h_l2": feature_sizes[0],
                    "g_l2": feature_sizes[1],
                    "local_l2": feature_sizes[2],
                    "c_l2": feature_sizes[3],
                    "Ag_g_l2": mean_channel_l2(transformed_g, mask),
                    "Al_l_l2": mean_channel_l2(transformed_l, mask),
                    "M_h_h_l2": contribution_sizes[0],
                    "M_g_g_l2": contribution_sizes[1],
                    "M_local_l_l2": contribution_sizes[2],
                    "M_c_c_l2": contribution_sizes[3],
                    "full_mixer_update_l2": mean_channel_l2(mixer_update, mask),
                    "coupling_share": contribution_sizes[3] / denominator if denominator > 0 else 0.0,
                }
                output_rows.append(row)
        return inspect_mixer

    try:
        # A forward pre-hook sees the input before each mixer acts on it.
        # The hook returns nothing, so the FULL forward pass is unchanged.
        for layer, mixer in enumerate(backbone.mix):
            hooks.append(mixer.register_forward_pre_hook(make_hook(layer)))
        tensor = torch.as_tensor(np.array(values, dtype=np.float64), dtype=torch.float64).view(1, n, 1)
        dt = torch.full((1,), dt_model, dtype=torch.float64)
        with torch.no_grad():
            prediction = model(tensor, dt)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Non-finite neural prediction")
    finally:
        # Always remove observation hooks before processing the next input.
        for hook in hooks:
            hook.remove()

    return max(reconstruction_errors) if reconstruction_errors else 0.0


def pure_convection_cases(n_samples: int, seed: int):
    """Randomly select independent conditions from both initial-data categories."""
    archive = torch.load(PURE_DATA, map_location="cpu", weights_only=False)
    inputs = archive["input"]
    metadata = archive.get("meta", {})
    types = metadata.get("ic_types")
    steps = int(metadata.get("n_steps_per_ic", 1))
    if types is None or len(types) != len(inputs):
        raise ValueError("PureConvection test data must include an ic_types entry for every pair")
    if steps <= 0:
        raise ValueError("Invalid n_steps_per_ic")

    # 5. Select independent initial conditions, not individual time pairs.
    # For PureConvection, sample smooth and piecewise conditions separately.
    by_category = defaultdict(list)
    for first_index in range(0, len(inputs), steps):
        category = str(types[first_index])
        if category in ("smooth_periodic", "piecewise_constant"):
            by_category[category].append(first_index)

    rng = np.random.default_rng(seed)
    grid = np.arange(inputs.shape[-1], dtype=np.float64) / inputs.shape[-1]
    # Analyze the initial profile and the snapshot at t=0.25 (when available).
    time_steps = sorted(set([0, min(5, steps - 1)]))
    counts = {}
    selected = []
    for category in ("smooth_periodic", "piecewise_constant"):
        choices = by_category[category]
        if not choices:
            raise ValueError(f"No PureConvection examples of type {category} were found")
        indexes = rng.choice(choices, size=min(n_samples, len(choices)), replace=False)
        counts[category] = len(indexes)
        selected.extend((int(index), category) for index in indexes)
    for first_index, category in sorted(selected):
        condition = first_index // steps + 1
        for time_step in time_steps:
            values = inputs[first_index + time_step].detach().cpu().numpy().astype(np.float64)
            # Smooth conditions use the whole domain; piecewise conditions
            # compare cells close to jumps with the remaining cells.
            if category == "smooth_periodic":
                regions = {"smooth": np.ones(len(grid), dtype=bool)}
            else:
                fronts = discontinuity_positions(values, grid)
                if not fronts:
                    raise RuntimeError(f"No discontinuity detected in PureConvection condition {condition}")
                near = periodic_near_mask(grid, fronts, radius_cells=4)
                regions = {"jump_near": near, "jump_far": ~near}
            yield values, regions, condition, float(time_step * metadata.get("dt", 0.05))
    print(f"[selection] PureConvection: {counts} independent conditions, seed={seed}, times {time_steps}")


def burgers_cases(n_samples: int, source: str, selected_files: list[Path], seed: int):
    """Choose reference snapshots before and after shock formation."""
    # For Burgers, read previously saved WENO-Z trajectories and front CSVs.
    # The optional "step" source is a separate discontinuous-data experiment.
    path = BURGERS_DATA if source == "main" else STEP_DATA
    fronts_path = BURGERS_FRONTS if source == "main" else STEP_FRONTS
    selected_files.append(fronts_path)
    positions = load_front_positions(fronts_path)
    first_shock_times = {}
    if source == "main":
        with fronts_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                first_shock_times[int(row["ic"]) - 1] = float(row["t_star"])
    with np.load(path, allow_pickle=False) as archive:
        grid = archive["x_low"].copy()
        times = archive["times"].copy()
        reference = archive["ref"]
        count = min(n_samples, len(reference))
        # Choose complete trajectories reproducibly, using the given seed.
        selected_conditions = sorted(np.random.default_rng(seed).choice(
            len(reference), size=count, replace=False).tolist())
        # The main experiment uses one pre-shock and three later snapshots.
        requested = [0.05, 0.5, 1.0, 4.0] if source == "main" else [0.0, 0.1, 0.3]
        selected_time_indices = []
        for time in requested:
            index = int(np.argmin(np.abs(times - time)))
            if np.isclose(times[index], time, rtol=0, atol=1e-10):
                selected_time_indices.append(index)
        if not selected_time_indices:
            raise ValueError("None of the requested times are available")
        for condition in selected_conditions:
            for time_index in selected_time_indices:
                time = float(times[time_index])
                values = reference[condition, time_index].copy()
                # Before the first theoretical shock, analyze the whole profile.
                # Afterwards, use the WENO-Z fronts to define near/far regions.
                if source == "main" and time <= 0.05 + 1e-10:
                    if condition not in first_shock_times or time >= first_shock_times[condition]:
                        raise ValueError(f"t={time:g} is not before the first theoretical shock for IC {condition + 1}")
                    regions = {"pre_shock_smooth": np.ones(len(grid), dtype=bool)}
                else:
                    key = (condition, round(time, 8))
                    if key not in positions:
                        raise ValueError(f"Missing WENO-Z fronts for IC {condition + 1}, t={time:g} in {fronts_path}")
                    near = periodic_near_mask(grid, positions[key], radius_cells=4)
                    regions = {"shock_near": near, "shock_far": ~near}
                yield values, regions, condition + 1, time
        print(f"[selection] Burgers ({source}): {count} conditions, seed={seed}, times {[float(times[i]) for i in selected_time_indices]}")


def summarize(rows: list[dict]) -> list[dict]:
    """Average within a layer and spatial region, weighting each analyzed input equally."""
    numeric_fields = [
        "h_l2", "g_l2", "local_l2", "c_l2", "Ag_g_l2", "Al_l_l2",
        "M_h_h_l2", "M_g_g_l2", "M_local_l_l2", "M_c_c_l2",
        "full_mixer_update_l2", "coupling_share",
    ]
    # 6. Build descriptive means for each layer and spatial region.
    # This file does not bootstrap; the next Task 3 script handles IC-level pairing.
    groups = defaultdict(list)
    for row in rows:
        groups[(row["problem"], row["source"], row["layer"], row["region"])].append(row)
    summaries = []
    for (problem, source, layer, region), group in sorted(groups.items()):
        summary = {"problem": problem, "source": source, "layer": layer,
                   "region": region, "n_cases": len(group)}
        for name in numeric_fields:
            summary[name] = float(np.mean([row[name] for row in group]))
        summaries.append(summary)
    return summaries


def plot_matrix_norms(rows: list[dict], path: Path) -> None:
    """Show Frobenius norms for each weight block and layer."""
    # 7. Plot fixed matrix sizes, layer by layer.
    names = ("M_h", "M_g", "M_local", "M_c", "A_g", "A_l")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in names:
        series = [row for row in rows if row["component"] == name]
        ax.plot([row["layer"] for row in series], [row["frobenius_norm"] for row in series], marker="o", label=name)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Frobenius norm")
    ax.set_xticks(sorted(set(row["layer"] for row in rows)))
    ax.grid(alpha=0.2)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_contributions(rows: list[dict], output_dir: Path) -> None:
    """Plot the coupling share and the four actual mixer contributions."""
    # Plot how actual branch magnitudes change with layer and spatial region.
    regions = sorted(set(row["region"] for row in rows))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for region in regions:
        subset = [row for row in rows if row["region"] == region]
        ax.plot([row["layer"] for row in subset],
                [row["coupling_share"] for row in subset], marker="o", label=region)
    ax.set_xlabel("Layer")
    ax.set_ylabel("||M_c c|| / sum of branch magnitudes")
    ax.set_xticks(sorted(set(row["layer"] for row in rows)))
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "coupling_share.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(regions), figsize=(max(7, 4 * len(regions)), 4.2), sharey=True)
    if len(regions) == 1:
        axes = [axes]
    fields = ("M_h_h_l2", "M_g_g_l2", "M_local_l_l2", "M_c_c_l2")
    for ax, region in zip(axes, regions):
        subset = [row for row in rows if row["region"] == region]
        for field in fields:
            ax.plot([row["layer"] for row in subset],
                    [row[field] for row in subset], marker="o", label=field)
        ax.set_title(region)
        ax.set_xlabel("Layer")
        ax.set_xticks(sorted(set(row["layer"] for row in rows)))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Mean spatial channel-L2 norm")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "branch_contributions.png", dpi=160)
    plt.close(fig)


def main() -> None:
    # 8. Read experiment settings; defaults reproduce the older small pilot.
    # Use --n-samples 50 and a new --outdir for the extended Report 1 analysis.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("PureConvection", "Burgers"), required=True)
    parser.add_argument("--source", choices=("main", "step"), default="main",
                        help="Use 'step' for the optional Burgers piecewise-constant test")
    parser.add_argument("--n-samples", type=int, default=6,
                        help="Independent conditions per category (or total for Burgers)")
    parser.add_argument("--seed", type=int, default=20261108,
                        help="Reproducible random selection of independent initial conditions")
    parser.add_argument("--threads", type=int, default=2,
                        help="Limit CPU thread contention on a laptop")
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    if args.n_samples < 1 or args.threads < 1:
        parser.error("--n-samples and --threads must be positive")
    if args.problem != "Burgers" and args.source != "main":
        parser.error("--source step is available only for Burgers")

    checkpoint_path = CHECKPOINTS[args.problem]
    data_path = PURE_DATA if args.problem == "PureConvection" else (BURGERS_DATA if args.source == "main" else STEP_DATA)
    front_path = None if args.problem == "PureConvection" else (BURGERS_FRONTS if args.source == "main" else STEP_FRONTS)
    for path in (checkpoint_path, data_path, front_path):
        if path is not None and not path.is_file():
            parser.error(f"Required file not found: {path}")

    torch.set_num_threads(args.threads)
    output_dir = args.outdir or (SCRIPT_DIR / "results" / args.problem / args.source)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the unchanged pretrained FULL model; do not train or ablate it.
    model, checkpoint = load_hybrid_step_map_1d(str(checkpoint_path), device="cpu")
    model.eval()
    if not hasattr(model, "backbone") or not hasattr(model.backbone, "mix"):
        raise TypeError("This script requires the original hybrid 1D backbone")
    if int(model.backbone.width) * 4 != model.backbone.mix[0].in_channels:
        raise ValueError("Unexpected mixer block layout")
    dt = float(checkpoint["dt"])
    print(f"[checkpoint] {args.problem} | dt={dt:g} | width={model.backbone.width} | layers={model.backbone.n_layers}")

    # Save static matrix measurements before inspecting any input profiles.
    weight_rows, singular_rows = matrix_statistics(args.problem, model.backbone)
    save_csv(output_dir / "matrix_norms.csv", weight_rows)
    save_csv(output_dir / "singular_values.csv", singular_rows)
    plot_matrix_norms(weight_rows, output_dir / "matrix_norms.png")

    if args.problem == "PureConvection":
        cases = pure_convection_cases(args.n_samples, args.seed)
    else:
        cases = burgers_cases(args.n_samples, args.source, [], args.seed)

    # Run each selected reference state through the FULL model and record
    # all four layer contributions in each requested spatial region.
    contribution_rows = []
    max_reconstruction_error = 0.0
    n_cases = 0
    for values, regions, condition, time in cases:
        error = analyze_input(model, values, regions, args.problem, args.source,
                              condition, time, dt, contribution_rows)
        max_reconstruction_error = max(max_reconstruction_error, error)
        n_cases += 1
        print(f"[analyzed] IC={condition} t={time:.2f} regions={','.join(regions)}")

    # Save per-case values, descriptive summaries, plots, and input provenance.
    # The separate task3_compare_branches.py performs paired near/far bootstrap.
    summary_rows = summarize(contribution_rows)
    save_csv(output_dir / "contributions_per_case.csv", contribution_rows)
    save_csv(output_dir / "contributions_summary.csv", summary_rows)
    plot_contributions(summary_rows, output_dir)

    fingerprints = {"checkpoint": file_hash(checkpoint_path), "data": file_hash(data_path)}
    if front_path is not None:
        fingerprints["front_csv"] = file_hash(front_path)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "problem": args.problem,
        "source": args.source,
        "n_samples_requested": args.n_samples,
        "selection_seed": args.seed,
        "n_inputs_analyzed": n_cases,
        "cpu_threads": args.threads,
        "torch_version": torch.__version__,
        "checkpoint_dt": dt,
        "checkpoint_path": str(checkpoint_path),
        "data_path": str(data_path),
        "front_csv_path": str(front_path) if front_path is not None else None,
        "sha256": fingerprints,
        "max_reconstruction_error": max_reconstruction_error,
        "method": (
            "Mean over spatial positions of the Euclidean L2 norm across 64 channels; "
            "one equal-weight contribution per analyzed input and spatial region. "
            "Coupling share is ||M_c c|| divided by the sum of four branch magnitudes. "
            "It is not an additive fraction of the full update. "
            "The mixer bias is included only in full_mixer_update_l2. "
            "Hooks inspect mixer inputs but never modify them."
        ),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[saved] {output_dir}")
    print(f"[check] Max reconstruction difference: {max_reconstruction_error:.3e}")
    print("[summary] Mean coupling share by region and layer:")
    for row in summary_rows:
        print(f"  {row['region']:17s} layer {row['layer']}: {row['coupling_share']:.5f} "
              f"(n={row['n_cases']})")
    print("Note: contributions are internal hidden-state vectors, not physical energy or flux.")


if __name__ == "__main__":
    main()

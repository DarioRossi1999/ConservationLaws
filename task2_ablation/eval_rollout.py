"""Task 2: compare FULL and NO-C LGNO rollouts with WENO-Z references.

Examples (run from the repository root):
    python task2_ablation/eval_rollout.py --problem PureConvection --T 3
    python task2_ablation/eval_rollout.py --problem Burgers --T 1
    python task2_ablation/eval_rollout.py --problem Burgers --T 4

Uses the original problem-specific IC generators, WENO-Z solvers and rollout
functions
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

# Locate the repository so this script can reuse the original model code.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ablation import without_coupling
from conslaw.checkpoints import load_hybrid_step_map_1d
from conslaw.eval_reports import save_rollout_reports

CHECKPOINTS = {
    "Burgers": "burgers_hybrid_flowmap_dt.pt",
    "PureConvection": "pureconvection_hybrid_flowmap_dt.pt",
}


def load_original_eval(problem):
    """Load the original evaluator without running its main function."""
    # Its functions generate initial conditions, compute WENO-Z references,
    # and advance LGNO in time. We reuse them instead of copying their code.
    folder = ROOT / problem
    path = folder / "eval.py"
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(f"_original_{problem}_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_per_ic_csv(path, times, l1_full, l1_noc, linf_full, linf_noc, ic_types):
    """Save final-time errors for each independent initial condition."""
    # Each CSV row describes one initial condition at the final time.
    # The delta columns measure NO-C error minus FULL error.
    columns = [
        "ic_index", "ic_type", "T", "rel_l1_final_full", "rel_l1_final_no_c",
        "delta_rel_l1_final", "rel_linf_final_full", "rel_linf_final_no_c",
        "delta_rel_linf_final",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for i, kind in enumerate(ic_types):
            writer.writerow({
                "ic_index": i,
                "ic_type": str(kind),
                "T": float(times[-1]),
                "rel_l1_final_full": float(l1_full[i, -1]),
                "rel_l1_final_no_c": float(l1_noc[i, -1]),
                "delta_rel_l1_final": float(l1_noc[i, -1] - l1_full[i, -1]),
                "rel_linf_final_full": float(linf_full[i, -1]),
                "rel_linf_final_no_c": float(linf_noc[i, -1]),
                "delta_rel_linf_final": float(linf_noc[i, -1] - linf_full[i, -1]),
            })


def main():
    # 1. Read the problem, simulation settings, and output folder from the CLI.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=CHECKPOINTS, required=True)
    parser.add_argument("--T", type=float, required=True)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--decay", type=float, default=2.0, help="Burgers Fourier decay")
    parser.add_argument("--nx-low", type=int, default=256)
    parser.add_argument("--upsample", type=int, default=4)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()

    if min(args.T, args.n_samples, args.nx_low, args.upsample, args.cfl) <= 0:
        parser.error("T, n-samples, nx-low, upsample and cfl must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    # 2. Load the pretrained LGNO and the original problem-specific evaluator.
    # No training or checkpoint modification happens in this script.
    torch.set_default_dtype(torch.float64)
    original = load_original_eval(args.problem)
    checkpoint = ROOT / args.problem / "checkpoints" / CHECKPOINTS[args.problem]
    model, info = load_hybrid_step_map_1d(str(checkpoint), device=torch.device(args.device))
    model.eval()
    if info.get("integrator") != "u_plus_dt_rhs":
        raise ValueError("Expected a dt-step model checkpoint")

    # Use the checkpoint time step and check that the requested final time fits.
    dt = float(info["dt"])
    n_steps = round(args.T / dt)
    if not np.isclose(n_steps * dt, args.T, rtol=0, atol=1e-10):
        raise ValueError("T must be a multiple of the checkpoint time step")
    if int(info["nx"]) != args.nx_low:
        raise ValueError("nx-low does not match the checkpoint grid")
    if not np.isclose(float(info.get("dx", 1 / args.nx_low)), 1 / args.nx_low,
                      rtol=0, atol=1e-12):
        raise ValueError("Grid spacing does not match the checkpoint")

    # Keep earlier results safe: refuse a destination that already has files.
    outdir = args.outdir or (ROOT / "task2_ablation" / "results" / args.problem /
                            "rollout" / f"T{args.T:g}_seed{args.seed}")
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {outdir}")

    # 3. Set up two grids: a fine grid for WENO-Z and a coarse grid for LGNO.
    # The random seed makes the sampled initial conditions reproducible.
    times = np.arange(n_steps + 1, dtype=np.float64) * dt
    nx_high = args.nx_low * args.upsample
    x_high = np.linspace(0, 1, nx_high, endpoint=False, dtype=np.float64)
    x_low = np.linspace(0, 1, args.nx_low, endpoint=False, dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    c = float(info.get("c", 1.0))
    bc = str(info.get("solver_bc", "periodic"))
    if args.problem == "Burgers" and bc != "periodic":
        raise ValueError("Burgers rollout requires periodic boundary conditions")

    # Store the error of every initial condition at every time step.
    shape = (args.n_samples, len(times))
    l1_full, l1_noc = np.zeros(shape), np.zeros(shape)
    linf_full, linf_noc = np.zeros(shape), np.zeros(shape)
    references, full_rollouts, noc_rollouts, fine_ics, ic_types = [], [], [], [], []

    # 4. Generate an initial condition and its full WENO-Z reference trajectory.
    for i in range(args.n_samples):
        if args.problem == "PureConvection":
            ic, kind = original.sample_random_ic(rng)
            u0_high = np.asarray(ic(x_high), dtype=np.float64)
            snapshots = original.solve_linear_advection_weno_ref(
                u0_high, times, 1 / nx_high, c=c, cfl=args.cfl
            )
        else:
            kind = "fourier_random"
            u0_high = original.random_fourier_ic(
                x_high, kmax=5, amp_range=(0.5, 1.0), decay=args.decay, rng=rng
            )
            snapshots = original.solve_burgers_weno_ref(
                u0_high, times, 1 / nx_high, bc=bc, cfl=args.cfl
            )

        # Average fine-grid cells to match the 256-point LGNO input grid.
        reference = np.stack([original.block_average(s, args.upsample) for s in snapshots])
        u0_low = reference[0]
        # 5. Starting from the same WENO-Z state, run LGNO twice.
        # FULL uses coupling; NO-C temporarily zeros it at each mixer input.
        # Later predictions are fed back into the model (free rollout).
        with torch.no_grad():
            full = original.rollout_hybrid_scalar(
                model, u0_low, n_steps, device, device.type == "cuda", dt_model=dt
            )
            with without_coupling(model):
                noc = original.rollout_hybrid_scalar(
                    model, u0_low, n_steps, device, device.type == "cuda", dt_model=dt
                )

        # Check that both rollouts start correctly and remain finite.
        if not np.array_equal(full[0], u0_low) or not np.array_equal(noc[0], u0_low):
            raise AssertionError("A rollout does not start from the reference initial state")
        if not (np.isfinite(full).all() and np.isfinite(noc).all()):
            raise FloatingPointError(f"Non-finite prediction for initial condition {i}")

        # 6. Compare each predicted state with WENO-Z at the same time.
        # L1 measures average spatial error; Linf measures the largest error.
        for j in range(len(times)):
            l1_full[i, j] = original.rel_l1(full[j], reference[j])
            l1_noc[i, j] = original.rel_l1(noc[j], reference[j])
            linf_full[i, j] = original.rel_linf(full[j], reference[j])
            linf_noc[i, j] = original.rel_linf(noc[j], reference[j])

        # Keep the complete trajectories for later plots and analyses.
        references.append(reference)
        full_rollouts.append(full)
        noc_rollouts.append(noc)
        fine_ics.append(u0_high)
        ic_types.append(kind)
        print(f"[{i + 1}/{args.n_samples}] {kind}: "
              f"final rel_L1 FULL={l1_full[i, -1]:.5g}, NO-C={l1_noc[i, -1]:.5g}", flush=True)

    # 7. Save summary statistics, error curves, and final errors per IC.
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "hybrid_full": {"l1": l1_full, "linf": linf_full},
        "hybrid_no_c": {"l1": l1_noc, "linf": linf_noc},
    }
    summary, curves = save_rollout_reports(outdir, times, metrics)
    per_ic = outdir / "per_ic_final.csv"
    save_per_ic_csv(per_ic, times, l1_full, l1_noc, linf_full, linf_noc, ic_types)

    # Save full trajectories and settings so they can be examined again
    # without rerunning WENO-Z or LGNO.
    archive = outdir / "trajectories_and_metrics.npz"
    np.savez_compressed(
        archive,
        times=times, x_low=x_low, x_high=x_high,
        ref=np.stack(references), full=np.stack(full_rollouts),
        no_c=np.stack(noc_rollouts), u0_fine=np.stack(fine_ics),
        ic_types=np.asarray(ic_types, dtype=str),
        l1_full=l1_full, l1_no_c=l1_noc,
        linf_full=linf_full, linf_no_c=linf_noc,
        seed=np.int64(args.seed), dt=np.float64(dt), cfl=np.float64(args.cfl),
        decay=np.float64(args.decay),
    )

    # 8. Print the average final errors and the paths of saved files.
    print("\n[final-time means]")
    print(f"  rel_L1   FULL={l1_full[:, -1].mean():.10g}  NO-C={l1_noc[:, -1].mean():.10g}")
    print(f"  rel_Linf FULL={linf_full[:, -1].mean():.10g}  NO-C={linf_noc[:, -1].mean():.10g}")
    print(f"  delta rel_L1 (NO-C - FULL) = {(l1_noc[:, -1] - l1_full[:, -1]).mean():+.10g}")
    print(f"  IC with larger NO-C error: {np.count_nonzero(l1_noc[:, -1] > l1_full[:, -1])}/{args.n_samples}")
    for path in (summary, curves, per_ic, archive):
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
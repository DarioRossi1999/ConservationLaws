"""Task 4: evaluate FULL, original NO-C and fine-tuned NO-C LGNO.

Reconstruct WENO-Z trajectories from ordered one-step test pairs.
Compare teacher-forced one-step errors, free rollouts and periodic TV.
Keep the CSV, figures and trajectory formats used in Report 1.

Example (from the repository root):
    python task4_finetuning/eval_no_c.py --problem PureConvection --outdir task4_finetuning/results/new_evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from conslaw.checkpoints import load_hybrid_step_map_1d

PROBLEMS = {
    "Burgers": {
        "test": "task4_finetuning/datasets/Burgers_200ic_seed20261110/test_pv.pt",
        "full": "Burgers/checkpoints/burgers_hybrid_flowmap_dt.pt",
        "finetuned": "task4_finetuning/results/Burgers/mixer_only_200ic_20epochs/best_no_c_mixer_finetuned.pt",
    },
    "PureConvection": {
        "test": "task4_finetuning/datasets/PureConvection_200ic_seed20261110/test_pv.pt",
        "full": "PureConvection/checkpoints/pureconvection_hybrid_flowmap_dt.pt",
        "finetuned": "task4_finetuning/results/PureConvection/mixer_only_200ic_20epochs/best_no_c_mixer_finetuned.pt",
    },
}


# The hook affects the forward pass only; the original weights stay unchanged.
@contextmanager
def without_coupling(model):
    """Temporarily zero the last 64 mixer input channels, without editing weights."""
    width = int(model.backbone.width)

    def zero_coupling(_module, inputs):
        (features,) = inputs
        if features.ndim != 3 or features.shape[1] != 4 * width:
            raise ValueError("Unexpected mixer input dimensions")
        modified = features.clone()
        modified[:, 3 * width:, :] = 0.0
        return (modified,)

    handles = [m.register_forward_pre_hook(zero_coupling) for m in model.backbone.mix]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def get_loss_function(problem: str):
    """Import the original PDE-specific loss rather than reimplementing it."""
    if problem == "Burgers":
        from Burgers.train import total_loss_batch
    else:
        from PureConvection.train import total_loss_batch
    return total_loss_batch


def save_csv(path: Path, records: list[dict]):
    if not records:
        raise ValueError(f"No CSV records for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def read_test_data(path: Path, problem: str, nx: int, dt: float):
    """Rebuild complete trajectories from ordered successive one-step pairs.

    Every group of consecutive pairs must satisfy input[t+1] == target[t].
    Reshaping pairs without checking this would silently invalidate rollouts.
    """
    # Read the one-step pairs and their time-grid metadata.
    archive = torch.load(path, map_location="cpu", weights_only=False)
    x = torch.as_tensor(archive["input"], dtype=torch.float64)
    y = torch.as_tensor(archive["output"], dtype=torch.float64)
    meta = archive["meta"]
    if x.ndim != 2 or x.shape != y.shape or x.shape[1] != nx:
        raise ValueError(f"Unexpected test data shape: {tuple(x.shape)}, {tuple(y.shape)}")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Non-finite values in the test archive")
    if int(meta["nx"]) != nx or abs(float(meta["dt"]) - dt) > 1e-12:
        raise ValueError("Dataset and checkpoint have different spatial/temporal grids")

    if problem == "Burgers":
        steps = int(meta["n_snaps"]) - 1
    else:
        if meta.get("equation") != "u_t + c u_x = 0":
            raise ValueError("Expected a PureConvection test archive")
        if abs(float(meta.get("c", float("nan"))) - 1.0) > 1e-12:
            raise ValueError("Expected advection speed c=1")
        steps = int(meta["n_steps_per_ic"])

    if steps < 1 or len(x) % steps:
        raise ValueError("Pairs cannot be divided into complete trajectories")
    # Group consecutive one-step pairs belonging to each initial condition.
    n_ic = len(x) // steps
    x_by_ic = x.reshape(n_ic, steps, nx)
    y_by_ic = y.reshape(n_ic, steps, nx)
    # Check that the output of one WENO-Z step is the input of the next.
    linking_error = (x_by_ic[:, 1:] - y_by_ic[:, :-1]).abs().max().item()
    if linking_error > 1e-10:
        raise ValueError(f"Pairs are not time-ordered (max discrepancy {linking_error:.3e})")
    # Rebuild complete WENO-Z trajectories, including the initial state.
    reference = torch.cat((x_by_ic[:, :1], y_by_ic), dim=1)
    times = np.arange(steps + 1, dtype=np.float64) * dt
    if "T" in meta and abs(float(meta["T"]) - times[-1]) > 1e-10:
        raise ValueError("Test metadata T disagrees with the time grid")

    types = None
    if problem == "PureConvection":
        labels = meta.get("ic_types")
        if not isinstance(labels, list) or len(labels) != len(x):
            raise ValueError("Expected one ic_types label per PureConvection pair")
        types = []
        for i in range(n_ic):
            subset = labels[i * steps:(i + 1) * steps]
            if len(set(subset)) != 1:
                raise ValueError(f"Condition {i} changes initial-condition type")
            types.append(subset[0])
    print(f"[data] {n_ic} initial conditions, {steps} steps each, T={times[-1]:g}")
    return x.unsqueeze(-1), y.unsqueeze(-1), reference, times, meta, linking_error, types


def check_checkpoint(problem: str, checkpoint: dict):
    if checkpoint.get("integrator") != "u_plus_dt_rhs":
        raise ValueError("Expected an LGNO dt-step checkpoint")
    if problem == "PureConvection":
        if "pureconvection" not in str(checkpoint.get("kind", "")).lower():
            raise ValueError("Expected a PureConvection checkpoint")
        if abs(float(checkpoint.get("c", float("nan"))) - 1.0) > 1e-12:
            raise ValueError("Expected c=1")


def verify_finetuned_checkpoint(original: dict, tuned: dict):
    """Check inactive weights for both fine-tuning modes without using hashes.

    A mixer-only checkpoint may change only the first three mixer blocks and
    mixer biases. An all-active checkpoint may update other functional layers,
    but the disabled coupling projections and mixer coupling block stay fixed.
    """
    # Check that fine-tuning did not change parameters that should be frozen.
    # The checkpoint still requires the NO-C hook during evaluation.
    info = tuned.get("task4_no_c")
    if not isinstance(info, dict) or not info.get("requires_without_coupling_hook"):
        raise ValueError("Fine-tuned checkpoint lacks the NO-C hook metadata")
    mode = info.get("fine_tuned")
    if mode not in ("mixers_only", "all_active"):
        raise ValueError(f"Unsupported fine-tuning mode: {mode}")
    a_weights, b_weights = original["model"], tuned["model"]
    if a_weights.keys() != b_weights.keys():
        raise ValueError("Checkpoint parameter names differ")
    largest_change = 0.0
    for name, before in a_weights.items():
        after = b_weights[name]
        if before.shape != after.shape:
            raise ValueError(f"Parameter shape changed: {name}")
        is_mixer_weight = name.startswith("backbone.mix.") and name.endswith(".weight")
        is_mixer_bias = name.startswith("backbone.mix.") and name.endswith(".bias")
        if is_mixer_weight:
            width = before.shape[0]
            if not torch.equal(before[:, 3 * width:, :], after[:, 3 * width:, :]):
                raise ValueError(f"Inactive coupling mixer block changed: {name}")
            largest_change = max(largest_change, (before[:, :3 * width, :] - after[:, :3 * width, :]).abs().max().item())
        elif mode == "mixers_only" and not is_mixer_bias:
            if not torch.equal(before, after):
                raise ValueError(f"Frozen parameter changed: {name}")
        elif mode == "all_active" and (name.startswith("backbone.w_g.") or name.startswith("backbone.w_l.")):
            if not torch.equal(before, after):
                raise ValueError(f"Inactive coupling projection changed: {name}")
        if is_mixer_bias:
            largest_change = max(largest_change, (before - after).abs().max().item())
    if mode == "mixers_only" and largest_change <= 0:
        raise ValueError("No active mixer parameter changed during fine-tuning")
    return mode, info, largest_change


def predict_step(model, x: torch.Tensor, dt: float):
    return model(x, x.new_full((len(x),), dt))


def relative_metrics(pred: np.ndarray, ref: np.ndarray):
    """Return per-profile absolute L1, relative L1 and relative Linf errors."""
    # Compute one error for each profile, before averaging over the test set.
    difference = np.abs(pred - ref)
    absolute_l1 = difference.mean(axis=-1)
    relative_l1 = absolute_l1 / (np.abs(ref).mean(axis=-1) + 1e-12)
    relative_linf = difference.max(axis=-1) / (np.abs(ref).max(axis=-1) + 1e-12)
    return absolute_l1, relative_l1, relative_linf


def total_variation_periodic(values: np.ndarray):
    """Sum absolute spatial jumps, including the periodic boundary jump."""
    # Include the jump between the last and first grid points (periodic domain).
    return np.abs(np.roll(values, -1, axis=-1) - values).sum(axis=-1)


@torch.no_grad()
def one_step_evaluation(model, x, y, dt, batch_size, no_c, loss_fn, device):
    """Evaluate using exact WENO-Z input at every step (teacher forcing)."""
    # One-step test: always predict from the exact WENO-Z input.
    model.eval()
    predictions = []
    batch_loss_sum = 0.0
    context = without_coupling(model) if no_c else nullcontext()
    with context:
        for start in range(0, len(x), batch_size):
            xx = x[start:start + batch_size].to(device)
            yy = y[start:start + batch_size].to(device)
            prediction = predict_step(model, xx, dt)
            loss, _, _, _, _ = loss_fn(
                prediction, yy, spec_mu=1.0, spec_kfrac=0.25,
                pred_spec_mu=0.0, lap_mu=0.0, model_bc="periodic",
            )
            # Weight each batch loss by its size, including a shorter last batch.
            batch_loss_sum += float(loss.item()) * len(xx)
            predictions.append(prediction.squeeze(-1).cpu().numpy())
    predicted = np.concatenate(predictions, axis=0)
    target = y.squeeze(-1).cpu().numpy()
    absolute, relative, relative_linf = relative_metrics(predicted, target)
    return {
        "loss": batch_loss_sum / len(x),
        "absolute_l1": float(absolute.mean()),
        "relative_l1": float(relative.mean()),
        "relative_linf": float(relative_linf.mean()),
        "per_pair_relative_l1": relative,
    }


@torch.no_grad()
def free_rollout(model, initial_states, steps, dt, batch_size, no_c, device):
    """Use each prediction as the input of the next time step."""
    # Free rollout: feed each predicted state into the following step.
    model.eval()
    predictions = []
    context = without_coupling(model) if no_c else nullcontext()
    with context:
        for start in range(0, len(initial_states), batch_size):
            u = initial_states[start:start + batch_size].unsqueeze(-1).to(device).clone()
            trajectory = [u.squeeze(-1).cpu().numpy()]
            for _ in range(steps):
                u = predict_step(model, u, dt)
                if not torch.isfinite(u).all():
                    raise FloatingPointError("Non-finite state encountered during rollout")
                trajectory.append(u.squeeze(-1).cpu().numpy())
            predictions.append(np.stack(trajectory, axis=1))
    return np.concatenate(predictions, axis=0)


def bootstrap_mean_interval(values: np.ndarray, seed: int, draws: int):
    """Resample independent initial conditions, never correlated time steps."""
    rng = np.random.default_rng(seed)
    # Each draw resamples whole initial conditions, not correlated time steps.
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    sampled = values[indices].mean(axis=1)
    return (float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975)))


def save_figures(output_dir, times, trajectories, reference, metrics, variants):
    """Save the same three figures as the two original Task 4A evaluators."""
    for title, label in zip(
        ("relative_l1", "relative_tv_error"),
        ("Mean relative L1 error", "Mean relative TV discrepancy"),
    ):
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for variant in variants:
            ax.plot(times, metrics[variant][title].mean(axis=0), label=variant.replace("_", " "))
        ax.set_xlabel("Time")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"rollout_{title}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(reference.shape[-1]) / reference.shape[-1]
    ax.plot(x, reference[0, -1], label="WENO-Z reference", linewidth=2.3)
    for variant in variants:
        ax.plot(x, trajectories[variant][0, -1], label=variant.replace("_", " "), alpha=0.85)
    ax.set_title(f"Test IC 0 at T={times[-1]:g}")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, T)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "profile_ic0_final.png", dpi=160)
    plt.close(fig)


def summarize_by_type(variants, ic_types, one_step_by_ic, final_records):
    """Provide the legacy PureConvection smooth/piecewise breakdown."""
    one_step_by_type, rollout_by_type = [], []
    for variant in variants:
        for label in sorted(set(ic_types)):
            n_type = ic_types.count(label)
            one_step_errors = [row["mean_relative_l1"] for row in one_step_by_ic
                               if row["variant"] == variant and row["ic_type"] == label]
            final_l1 = [row["relative_l1"] for row in final_records
                        if row["variant"] == variant and row["ic_type"] == label]
            final_tv = [row["relative_tv_error"] for row in final_records
                        if row["variant"] == variant and row["ic_type"] == label]
            one_step_by_type.append({
                "variant": variant, "ic_type": label, "n_conditions": n_type,
                "mean_relative_l1": float(np.mean(one_step_errors)),
            })
            rollout_by_type.append({
                "variant": variant, "ic_type": label, "n_conditions": n_type,
                "mean_relative_l1": float(np.mean(final_l1)),
                "mean_relative_tv_error": float(np.mean(final_tv)),
            })
    return one_step_by_type, rollout_by_type


def check_output_dir(output_dir: Path, input_files: list[Path]):
    """Disallow overwriting the paper's original code or existing results."""
    output_dir = output_dir.resolve()
    protected = [ROOT / name for name in ("Burgers", "PureConvection", "src", "task2_ablation")]
    if any(output_dir == folder.resolve() or folder.resolve() in output_dir.parents for folder in protected):
        raise ValueError("Outputs must not be stored inside original source folders")
    if any(output_dir == file.resolve() or output_dir in file.resolve().parents for file in input_files):
        raise ValueError("Output directory would contain an input file")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, choices=list(PROBLEMS))
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--full-checkpoint", type=Path)
    parser.add_argument("--finetuned-checkpoint", type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--bootstrap-draws", type=int, default=0,
                        help="Set 2000 to reproduce the historical bootstrap intervals; 0 disables them")
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.threads <= 0 or args.bootstrap_draws < 0:
        parser.error("Batch size and threads must be positive; bootstrap draws must be nonnegative")
    return args


def main():
    args = parse_args()
    settings = PROBLEMS[args.problem]
    test_file = args.test_file or ROOT / settings["test"]
    full_file = args.full_checkpoint or ROOT / settings["full"]
    tuned_file = args.finetuned_checkpoint or ROOT / settings["finetuned"]
    for path in (test_file, full_file, tuned_file):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")
    output_dir = check_output_dir(args.outdir, [test_file, full_file, tuned_file])

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # 1. Load the original and fine-tuned checkpoints, then verify compatibility.
    full_model, full_ckpt = load_hybrid_step_map_1d(str(full_file), device=device)
    tuned_model, tuned_ckpt = load_hybrid_step_map_1d(str(tuned_file), device=device)
    check_checkpoint(args.problem, full_ckpt)
    check_checkpoint(args.problem, tuned_ckpt)
    dt = float(full_ckpt["dt"])
    nx = int(full_ckpt["nx"])
    if int(tuned_ckpt["nx"]) != nx or abs(float(tuned_ckpt["dt"]) - dt) > 1e-12:
        raise ValueError("Checkpoint spatial/temporal grids differ")
    fine_tuning_mode, training_info, max_mixer_change = verify_finetuned_checkpoint(full_ckpt, tuned_ckpt)
    variant_ft = "no_c_mixer_finetuned" if fine_tuning_mode == "mixers_only" else "no_c_all_active_finetuned"
    variants = ("full_original", "no_c_original", variant_ft)
    print(f"[model] {args.problem}, fine_tuning={fine_tuning_mode}, max_mixer_change={max_mixer_change:.3e}")
    x, y, reference_tensor, times, test_meta, linking_error, ic_types = read_test_data(test_file, args.problem, nx, dt)
    reference = reference_tensor.numpy()
    n_ic, n_times, _ = reference.shape
    steps = n_times - 1
    variant_models = {
        "full_original": (full_model, False),
        "no_c_original": (full_model, True),
        variant_ft: (tuned_model, True),
    }
    loss_fn = get_loss_function(args.problem)

    # 2. Evaluate the same test pairs for all three model variants.
    one_step_records = []
    one_step_by_ic = []
    for variant, (model, no_c) in variant_models.items():
        result = one_step_evaluation(model, x, y, dt, args.batch_size, no_c, loss_fn, device)
        one_step_records.append({
            "variant": variant, "n_conditions": n_ic, "n_pairs": len(x),
            "loss": result["loss"], "absolute_l1": result["absolute_l1"],
            "relative_l1": result["relative_l1"], "relative_linf": result["relative_linf"],
        })
        errors = result["per_pair_relative_l1"].reshape(n_ic, steps)
        for ic in range(n_ic):
            row = {"variant": variant, "ic_index": ic}
            if ic_types is not None:
                row["ic_type"] = ic_types[ic]
            row["mean_relative_l1"] = float(errors[ic].mean())
            one_step_by_ic.append(row)
        print(f"[one step] {variant}: loss={result['loss']:.8g}, relative L1={result['relative_l1']:.8g}")

    # 3. Start every free rollout from the same exact WENO-Z initial state.
    trajectories, metrics = {}, {}
    tv_reference = total_variation_periodic(reference)
    for variant, (model, no_c) in variant_models.items():
        predicted = free_rollout(model, reference_tensor[:, 0], steps, dt,
                                 args.batch_size, no_c, device)
        trajectories[variant] = predicted
        absolute, relative, relative_linf = relative_metrics(predicted, reference)
        tv_pred = total_variation_periodic(predicted)
        # TV is compared per initial condition; matching average TV is not enough.
        metrics[variant] = {
            "absolute_l1": absolute,
            "relative_l1": relative,
            "relative_linf": relative_linf,
            "pred_tv": tv_pred,
            "relative_tv_error": np.abs(tv_pred - tv_reference) / (tv_reference + 1e-12),
        }
        print(f"[rollout] {variant}: relative L1(T)={relative[:, -1].mean():.8g}, "
              f"relative TV(T)={metrics[variant]['relative_tv_error'][:, -1].mean():.8g}")

    # 4. Prepare time-resolved errors and final-time errors for each IC.
    curve_records, final_records, final_summary = [], [], []
    for variant in variants:
        data = metrics[variant]
        for j, t in enumerate(times):
            curve_records.append({
                "variant": variant, "time": float(t), "n_conditions": n_ic,
                "mean_absolute_l1": float(data["absolute_l1"][:, j].mean()),
                "mean_relative_l1": float(data["relative_l1"][:, j].mean()),
                "mean_relative_linf": float(data["relative_linf"][:, j].mean()),
                "mean_reference_tv": float(tv_reference[:, j].mean()),
                "mean_predicted_tv": float(data["pred_tv"][:, j].mean()),
                "mean_relative_tv_error": float(data["relative_tv_error"][:, j].mean()),
            })
        for ic in range(n_ic):
            row = {"variant": variant, "ic_index": ic}
            if ic_types is not None:
                row["ic_type"] = ic_types[ic]
            row.update({
                "final_time": float(times[-1]),
                "relative_l1": float(data["relative_l1"][ic, -1]),
                "relative_linf": float(data["relative_linf"][ic, -1]),
                "absolute_l1": float(data["absolute_l1"][ic, -1]),
                "reference_tv": float(tv_reference[ic, -1]),
                "predicted_tv": float(data["pred_tv"][ic, -1]),
                "relative_tv_error": float(data["relative_tv_error"][ic, -1]),
            })
            final_records.append(row)
        final_summary.append({
            "variant": variant, "n_conditions": n_ic, "final_time": float(times[-1]),
            "mean_relative_l1": float(data["relative_l1"][:, -1].mean()),
            "mean_relative_linf": float(data["relative_linf"][:, -1].mean()),
            "mean_absolute_l1": float(data["absolute_l1"][:, -1].mean()),
            "mean_relative_tv_error": float(data["relative_tv_error"][:, -1].mean()),
            "mean_reference_tv": float(tv_reference[:, -1].mean()),
            "mean_predicted_tv": float(data["pred_tv"][:, -1].mean()),
        })

    # 5. Compare final-time errors on paired initial conditions.
    final_noc = metrics["no_c_original"]["relative_l1"][:, -1]
    final_ft = metrics[variant_ft]["relative_l1"][:, -1]
    final_full = metrics["full_original"]["relative_l1"][:, -1]
    paired = {
        "no_c_original_minus_finetuned": {
            "mean": float(np.mean(final_noc - final_ft)),
            "number_improved": int(np.count_nonzero(final_ft < final_noc)),
        },
        "finetuned_minus_full": {
            "mean": float(np.mean(final_ft - final_full)),
        },
    }
    if args.bootstrap_draws:
        paired["no_c_original_minus_finetuned"]["bootstrap_95_percent"] = bootstrap_mean_interval(
            final_noc - final_ft, args.seed, args.bootstrap_draws
        )
        paired["finetuned_minus_full"]["bootstrap_95_percent"] = bootstrap_mean_interval(
            final_ft - final_full, args.seed + 1, args.bootstrap_draws
        )

    # 6. Save all legacy CSVs, trajectories, figures and experiment metadata.
    output_dir.mkdir(parents=True, exist_ok=False)
    if ic_types is not None:
        one_by_type, final_by_type = summarize_by_type(variants, ic_types, one_step_by_ic, final_records)
        save_csv(output_dir / "one_step_by_type.csv", one_by_type)
        save_csv(output_dir / "rollout_final_by_type.csv", final_by_type)
    save_csv(output_dir / "one_step_summary.csv", one_step_records)
    save_csv(output_dir / "one_step_per_ic.csv", one_step_by_ic)
    save_csv(output_dir / "rollout_curves.csv", curve_records)
    save_csv(output_dir / "rollout_final_per_ic.csv", final_records)
    save_csv(output_dir / "rollout_final_summary.csv", final_summary)
    np.savez_compressed(output_dir / "test_trajectories.npz", times=times, reference=reference,
                        **trajectories)
    save_figures(output_dir, times, trajectories, reference, metrics, variants)
    metadata = {
        "task": f"Task 4A {args.problem} one-step and rollout evaluation",
        "n_test_conditions": n_ic,
        "n_pairs": len(x),
        "n_rollout_steps": steps,
        "dt": dt,
        "test_metadata": test_meta,
        "max_pair_linking_error": linking_error,
        "max_active_weight_change": max_mixer_change,
        "finetuned_checkpoint_metadata": training_info,
        "metric_definition": {
            "relative_l1": "mean(abs(pred-ref))/(mean(abs(ref))+1e-12), per profile",
            "relative_linf": "max(abs(pred-ref))/(max(abs(ref))+1e-12), per profile",
            "tv": "sum(abs(roll(u,-1)-u)) on the periodic grid",
            "relative_tv_error": "abs(TV(pred)-TV(ref))/(TV(ref)+1e-12)",
            "one_step_loss": f"{args.problem} original L1 + spectral(mu=1, kfrac=0.25)",
            "aggregation": "mean over independent IC; one-step mean over one-step pairs",
            "checkpoint_selection": "Best validation loss, never test loss",
        },
        "paired_final_relative_l1": paired,
    }
    if ic_types is not None:
        metadata["test_ic_types"] = ic_types
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    print(f"[saved] {output_dir}")
    print("[note] Use validation data, not this test split, for selecting hyperparameters")


if __name__ == "__main__":
    main()

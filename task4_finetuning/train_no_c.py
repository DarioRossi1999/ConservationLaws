
"""Fine-tune a pretrained 1D LGNO without multiplicative coupling.

The same trainer supports Burgers and PureConvection, mixer-only or all-active
fine-tuning, CPU/GPU execution, and resumable epoch-boundary checkpoints.

Compatibility with Task 4A is obtained by passing its original data files,
seed, batch size, optimizer settings, epoch count, and update limit.
The original repository, datasets, and pretrained checkpoints are never written.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from conslaw.checkpoints import load_hybrid_step_map_1d

PROBLEMS = {
    "Burgers": {
        "checkpoint": "Burgers/checkpoints/burgers_hybrid_flowmap_dt.pt",
        "train": "Burgers/dataset/train_pv.pt",
        "validation": "Burgers/dataset/val_pv.pt",
        "equation": "u_t + (u^2/2)_x = 0",
    },
    "PureConvection": {
        "checkpoint": "PureConvection/checkpoints/pureconvection_hybrid_flowmap_dt.pt",
        "train": "PureConvection/dataset/train_pv.pt",
        "validation": "PureConvection/dataset/val_pv.pt",
        "equation": "u_t + c u_x = 0",
    },
}


def read_pairs(path: Path, limit: int | None, seed: int):
    """Load one-step pairs and retain the Task 4A random-selection order.

    Pairs from the same physical trajectory are not independent observations.
    """
    archive = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(archive, dict) or "input" not in archive or "output" not in archive:
        raise ValueError(f"Expected 'input' and 'output' tensors in {path}")
    inputs, targets = archive["input"], archive["output"]
    if inputs.ndim == 2:
        inputs = inputs.unsqueeze(-1)
    if targets.ndim == 2:
        targets = targets.unsqueeze(-1)
    if inputs.shape != targets.shape or inputs.ndim != 3 or inputs.shape[-1] != 1:
        raise ValueError(f"Unexpected pair shapes in {path}: {tuple(inputs.shape)}, {tuple(targets.shape)}")
    if len(inputs) == 0:
        raise ValueError(f"Empty dataset: {path}")

    # Use the same seeded pair selection as the Task 4A CPU experiments.
    # Even if all pairs are selected, keep their original random order.
    count = len(inputs) if limit is None else min(limit, len(inputs))
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(len(inputs), generator=generator)[:count]
    x = inputs[selected].to(dtype=torch.float64).contiguous()
    y = targets[selected].to(dtype=torch.float64).contiguous()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError(f"Non-finite data in {path}")
    return x, y, archive.get("meta", {})


def validate_data(problem: str, checkpoint: dict, train_meta: dict, val_meta: dict,
                  x_train: torch.Tensor, x_val: torch.Tensor) -> tuple[float, int]:
    """Check that training and validation use the pretrained model's PDE and grid."""
    if checkpoint.get("integrator") != "u_plus_dt_rhs":
        raise ValueError("Expected a dt-step checkpoint")
    if checkpoint.get("bc", "periodic") != "periodic":
        raise ValueError("Only periodic checkpoints are supported")
    if problem.lower() not in str(checkpoint.get("kind", "")).lower():
        raise ValueError(f"Checkpoint kind does not match {problem}")
    dt = float(checkpoint["dt"])
    nx = int(checkpoint["nx"])
    if x_train.shape[1] != nx or x_val.shape[1] != nx:
        raise ValueError("Dataset grid size differs from the checkpoint")
    for meta in (train_meta, val_meta):
        if "equation" in meta and meta["equation"] != PROBLEMS[problem]["equation"]:
            raise ValueError(f"Dataset equation does not match {problem}")
        if "dt" in meta and abs(float(meta["dt"]) - dt) > 1e-12:
            raise ValueError("Dataset time step differs from the checkpoint")
        if "nx" in meta and int(meta["nx"]) != nx:
            raise ValueError("Dataset grid size differs from the checkpoint")
        if problem == "PureConvection":
            if "c" not in meta or "c" not in checkpoint:
                raise ValueError("Advection speed is missing")
            if abs(float(meta["c"]) - float(checkpoint["c"])) > 1e-12:
                raise ValueError("Advection speed differs from the checkpoint")
    return dt, nx


@contextmanager
def without_coupling(model):
    """Zero the coupling channels immediately before each mixer forward pass.

    The hook changes the mixer's input, not its weights or the saved checkpoint.
    Removing every hook in ``finally`` makes the intervention reversible.
    """
    width = int(model.backbone.width)

    def zero_coupling(_module, inputs):
        (features,) = inputs
        if features.ndim != 3 or features.shape[1] != 4 * width:
            raise ValueError("Unexpected mixer input shape")
        modified = features.clone()
        modified[:, 3 * width:, :] = 0.0
        return (modified,)

    hooks = [mixer.register_forward_pre_hook(zero_coupling) for mixer in model.backbone.mix]
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


def configure_trainable(model, mode: str):
    """Select active parameters and freeze the absent coupling pathways.

    Mixer weights remain full-sized tensors for checkpoint compatibility.
    Their inactive last input block is gradient-masked and restored after
    every optimizer step, even when AdamW weight decay is nonzero.
    """
    backbone = model.backbone
    width = int(backbone.width)
    # Mixer-only updates the three active mixer blocks and their biases.
    # All-active also trains the remaining functional layers of the network.
    if mode == "mixer-only":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        # These two projections are used only to compute the disabled coupling.
        for module in (*backbone.w_g, *backbone.w_l):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    # The last mixer block multiplies the disabled coupling channels.
    # Mask its gradients and save its original weights so that AdamW's
    # weight decay cannot modify this inactive block.
    coupling_snapshots = []
    for mixer in backbone.mix:
        if not isinstance(mixer, torch.nn.Conv1d) or mixer.weight.shape != (width, 4 * width, 1):
            raise TypeError("Unexpected mixer structure")
        mixer.weight.requires_grad_(True)
        mixer.bias.requires_grad_(True)
        mask = torch.ones_like(mixer.weight)
        mask[:, 3 * width:, :] = 0.0
        mixer.weight.register_hook(lambda grad, fixed_mask=mask: grad * fixed_mask)
        coupling_snapshots.append(mixer.weight[:, 3 * width:, :].detach().clone())

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    # Exclude inactive mixer columns when counting effective trainable weights.
    effective = sum(parameter.numel() for parameter in trainable) - len(backbone.mix) * width * width
    return trainable, coupling_snapshots, effective


@torch.no_grad()
def restore_coupling_weights(model, originals):
    """Prevent decoupled weight decay from changing inactive mixer columns."""
    width = int(model.backbone.width)
    for mixer, initial in zip(model.backbone.mix, originals):
        mixer.weight[:, 3 * width:, :].copy_(initial)


def get_loss_function(problem: str):
    """Use the original repository implementation to avoid changing the loss."""
    if problem == "Burgers":
        from Burgers.train import total_loss_batch
    else:
        from PureConvection.train import total_loss_batch
    return total_loss_batch


def loss_components(loss_fn, prediction, target, spec_mu: float):
    # Keep the original PDE-specific MAE + high-frequency spectral loss.
    total, absolute_l1, _, _, _ = loss_fn(
        prediction, target,
        spec_mu=spec_mu, spec_kfrac=0.25,
        pred_spec_mu=0.0, lap_mu=0.0, model_bc="periodic",
    )
    return total, absolute_l1


def predict(model, x: torch.Tensor, dt: float):
    return model(x, x.new_full((len(x),), dt))


@torch.no_grad()
def validate(model, inputs, targets, batch_size, dt, spec_mu, loss_fn, device, *, no_c):
    """Average each relative L1 error per profile before averaging profiles."""
    model.eval()
    # Average per-example errors over all validation pairs, including
    # the final batch if it is smaller than the requested batch size.
    loss_sum = l1_sum = relative_sum = 0.0
    context = without_coupling(model) if no_c else nullcontext()
    with context:
        for start in range(0, len(inputs), batch_size):
            x = inputs[start:start + batch_size].to(device)
            target = targets[start:start + batch_size].to(device)
            prediction = predict(model, x, dt)
            total, absolute_l1 = loss_components(loss_fn, prediction, target, spec_mu)
            relative = (prediction - target).abs().mean(dim=(1, 2)) / (
                target.abs().mean(dim=(1, 2)) + 1e-12
            )
            loss_sum += total.item() * len(x)
            l1_sum += absolute_l1.item() * len(x)
            relative_sum += relative.sum().item()
    count = len(inputs)
    return {
        "loss": loss_sum / count,
        "absolute_l1": l1_sum / count,
        "relative_l1": relative_sum / count,
    }


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_best(path: Path, model, original_checkpoint: dict, metadata: dict):
    """Keep the original checkpoint schema for the existing evaluation scripts."""
    # Save the trained model in the same format as the pretrained checkpoint.
    # The evaluator will still need the NO-C hook when loading these weights.
    checkpoint = dict(original_checkpoint)
    checkpoint["model"] = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    checkpoint["task4_no_c"] = metadata
    torch.save(checkpoint, path)


def rng_snapshot(shuffle_generator: torch.Generator):
    """Save random states to reproduce the next epoch after a resume."""
    state = {
        "shuffle_generator": shuffle_generator.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(snapshot: dict, shuffle_generator: torch.Generator):
    shuffle_generator.set_state(snapshot["shuffle_generator"])
    torch.set_rng_state(snapshot["torch_cpu"])
    np.random.set_state(snapshot["numpy"])
    random.setstate(snapshot["python"])
    if torch.cuda.is_available() and "torch_cuda" in snapshot:
        torch.cuda.set_rng_state_all(snapshot["torch_cuda"])


def save_last(path: Path, *, model, optimizer, scheduler, epoch, updates,
              best_loss, best_epoch, stale_epochs, history, baselines, shuffle_generator, config):
    """Atomically write a restartable state after a complete validation epoch."""
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "updates": updates,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "history": history,
        "baselines": baselines,
        "rng": rng_snapshot(shuffle_generator),
        "config": config,
    }
    # Write a temporary file first to avoid leaving a partial resume state.
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def make_optimizer(args, parameters):
    optimizer_class = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
    return optimizer_class(parameters, lr=args.lr, weight_decay=args.weight_decay)


def make_scheduler(args, optimizer):
    if args.schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return None


def check_output_dir(args, directory: Path, input_paths: list[Path]):
    """Avoid overwriting sources or silently mixing independent experiment runs."""
    directory = directory.resolve()
    if directory == ROOT.resolve() or ROOT.resolve() not in directory.parents:
        raise ValueError(f"Output must be a new directory under the repository: {ROOT}")
    protected = [ROOT / name for name in ("Burgers", "PureConvection", "src", "task2_ablation", "task3_analysis")]
    if any(directory == path.resolve() or path.resolve() in directory.parents for path in protected):
        raise ValueError("Output directory overlaps original sources or prior analysis tasks")
    if any(directory == source.resolve() or directory in source.resolve().parents for source in input_paths):
        raise ValueError("Source checkpoint or dataset is inside the output directory")
    if args.resume is None and directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Output directory is not empty: {directory}")
    if args.resume is not None and args.resume.resolve().parent != directory:
        raise ValueError("The resume state must reside in the requested output directory")


def run_benchmark(args, model, optimizer, originals, x_train, y_train, loss_fn, dt, device, output_dir):
    """Perform a short destructive-in-memory benchmark without saving model weights."""
    # The benchmark times real optimization steps but does not save weights.
    # Always launch it as a separate process, not before a training run.
    model.train()
    x = x_train[:args.batch_size].to(device)
    y = y_train[:args.batch_size].to(device)
    durations = []
    with without_coupling(model):
        for step in range(args.benchmark_steps):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            prediction = predict(model, x, dt)
            loss, _ = loss_components(loss_fn, prediction, y, args.spec_mu)
            loss.backward()
            optimizer.step()
            restore_coupling_weights(model, originals)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - start)
            print(f"[benchmark] step={step + 1} loss={loss.item():.8g} time={durations[-1]:.3f}s")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_results.json").write_text(json.dumps({
        "device": str(device), "batch_size": len(x), "seconds_per_step": durations,
        "mean_seconds_per_step": float(np.mean(durations)),
    }, indent=2), encoding="utf-8")


def experiment_config(args, source_paths):
    """Record settings needed to distinguish experiments and validate a resume."""
    return {
        "problem": args.problem,
        "trainable": args.trainable,
        "optimizer": args.optimizer,
        "schedule": args.schedule,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "spec_mu": args.spec_mu,
        "batch_size": args.batch_size,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "seed": args.seed,
        "checkpoint": str(source_paths["checkpoint"].resolve()),
        "train_file": str(source_paths["train"].resolve()),
        "val_file": str(source_paths["validation"].resolve()),
        "epochs": args.epochs,
        "max_updates": args.max_updates,
    }


def run_training(args, model, optimizer, scheduler, coupling_weights, original_checkpoint,
                 x_train, y_train, x_val, y_val, loss_fn, dt, device, output_dir, config):
    history_path = output_dir / "training_history.csv"
    summary_path = output_dir / "validation_summary.csv"
    best_path = output_dir / "best_no_c_mixer_finetuned.pt" if args.trainable == "mixer-only" else output_dir / "best_no_c_all_active_finetuned.pt"
    last_path = output_dir / "last_training_state.pt"

    generator = torch.Generator().manual_seed(args.seed + 10)
    history = []
    updates = 0
    best_epoch = 0
    start_epoch = 1
    best_loss = float("inf")
    stale_epochs = 0

    if args.resume is not None:
        # Restore model, optimizer, scheduler and random states together.
        # A resume starts after the last fully recorded epoch.
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state["config"] != config:
            raise ValueError("Resume settings differ from the saved experiment configuration")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state["scheduler"] is not None:
            scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"], generator)
        history = list(state["history"])
        updates = int(state["updates"])
        best_loss, best_epoch = float(state["best_loss"]), int(state["best_epoch"])
        stale_epochs = int(state.get("stale_epochs", 0))
        start_epoch = int(state["epoch"]) + 1
        baseline_full, baseline_no_c = state["baselines"]
        print(f"[resume] start_epoch={start_epoch} completed_updates={updates}")
    else:
        # Measure FULL and NO-C before changing any pretrained parameters.
        baseline_full = validate(model, x_val, y_val, args.batch_size, dt, args.spec_mu,
                                 loss_fn, device, no_c=False)
        baseline_no_c = validate(model, x_val, y_val, args.batch_size, dt, args.spec_mu,
                                 loss_fn, device, no_c=True)
        print(f"[baseline FULL] {baseline_full}")
        print(f"[baseline NO-C] {baseline_no_c}")

    elapsed = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        # Shuffle the same selected training pairs once per epoch.
        model.train()
        shuffled = torch.randperm(len(x_train), generator=generator)
        epoch_losses = []
        epoch_start = time.perf_counter()

        # Fine-tune with coupling disabled at every mixer input.
        with without_coupling(model):
            for start in range(0, len(shuffled), args.batch_size):
                if args.max_updates is not None and updates >= args.max_updates:
                    break
                indices = shuffled[start:start + args.batch_size]
                x = x_train[indices].to(device)
                target = y_train[indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = predict(model, x, dt)
                loss, _ = loss_components(loss_fn, prediction, target, args.spec_mu)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                # Restore inactive mixer weights after Adam/AdamW updates.
                restore_coupling_weights(model, coupling_weights)
                epoch_losses.append(loss.item())
                updates += 1

        if not epoch_losses:
            break
        # Validate with NO-C after every epoch, before choosing a checkpoint.
        metrics = validate(model, x_val, y_val, args.batch_size, dt, args.spec_mu,
                           loss_fn, device, no_c=True)
        row = {
            "epoch": epoch,
            "updates": updates,
            "training_loss_mean": float(np.mean(epoch_losses)),
            "validation_loss": metrics["loss"],
            "validation_absolute_l1": metrics["absolute_l1"],
            "validation_relative_l1": metrics["relative_l1"],
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        if scheduler is not None:
            scheduler.step()

        # Choose the best checkpoint using validation loss, never test loss.
        if metrics["loss"] < best_loss:
            relative_improvement = (best_loss - metrics["loss"]) / max(abs(best_loss), 1e-12) if math.isfinite(best_loss) else float("inf")
            best_loss, best_epoch = metrics["loss"], epoch
            stale_epochs = 0 if relative_improvement >= args.min_relative_improvement else stale_epochs + 1
            save_best(best_path, model, original_checkpoint, {
                "no_c": True, "requires_without_coupling_hook": True,
                "fine_tuned": "mixers_only" if args.trainable == "mixer-only" else "all_active",
                "seed": args.seed, "epoch": epoch, "updates": updates,
                "best_validation_loss": best_loss,
                "training_script": "task4_finetuning/train_no_c.py",
            })
            print(f"[best] epoch={epoch} val_loss={best_loss:.8g}")
        else:
            stale_epochs += 1

        # Preserve Task 4A CSV columns and save a restart point each epoch.
        save_csv(history_path, history, list(history[0]))
        save_last(last_path, model=model, optimizer=optimizer, scheduler=scheduler,
                  epoch=epoch, updates=updates, best_loss=best_loss, best_epoch=best_epoch,
                  stale_epochs=stale_epochs, history=history, baselines=(baseline_full, baseline_no_c),
                  shuffle_generator=generator, config=config)
        print(f"[epoch {epoch:03d}] updates={updates} train={row['training_loss_mean']:.6g} "
              f"val={metrics['loss']:.6g} rel_l1={metrics['relative_l1']:.6g} "
              f"lr={optimizer.param_groups[0]['lr']:.3g}")
        if args.patience and epoch >= args.min_epochs and stale_epochs >= args.patience:
            print(f"[early stop] No meaningful improvement for {stale_epochs} epochs")
            break
        if args.max_updates is not None and updates >= args.max_updates:
            break

    if not best_path.is_file():
        raise RuntimeError("Training produced no best checkpoint")
    # Evaluate the best checkpoint, which may come from an earlier epoch.
    best_model, _ = load_hybrid_step_map_1d(str(best_path), device=device)
    best_metrics = validate(best_model, x_val, y_val, args.batch_size, dt, args.spec_mu,
                            loss_fn, device, no_c=True)
    save_csv(summary_path, [
        {"variant": "full_original", **baseline_full},
        {"variant": "no_c_original", **baseline_no_c},
        {"variant": "no_c_mixer_finetuned_best" if args.trainable == "mixer-only"
         else "no_c_all_active_finetuned_best", **best_metrics},
    ], ["variant", "loss", "absolute_l1", "relative_l1"])
    (output_dir / "run_metadata.json").write_text(json.dumps({
        "settings": config, "training_updates": updates, "best_epoch": best_epoch,
        "elapsed_training_seconds": time.perf_counter() - elapsed,
        "note": "Use the NO-C mixer hook when loading this checkpoint for evaluation.",
    }, indent=2), encoding="utf-8")
    print(f"[done] best_epoch={best_epoch} updates={updates} checkpoint={best_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, choices=list(PROBLEMS))
    parser.add_argument("--mode", choices=("train", "benchmark"), default="train")
    parser.add_argument("--trainable", choices=("mixer-only", "all-active"), default="mixer-only")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--val-file", type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--spec-mu", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--benchmark-steps", type=int, default=5)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--min-relative-improvement", type=float, default=0.005)
    args = parser.parse_args()
    for name in ("batch_size", "epochs", "threads", "benchmark_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("n_train", "n_val", "max_updates"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or args.spec_mu < 0:
        parser.error("Learning rate must be positive; decay and spectral weight nonnegative")
    if args.patience < 0 or args.min_epochs < 1 or args.min_relative_improvement < 0:
        parser.error("Invalid early-stopping settings")
    if args.mode == "benchmark" and args.resume is not None:
        parser.error("Cannot resume a benchmark")
    return args


def main():
    args = parse_args()
    settings = PROBLEMS[args.problem]
    checkpoint_path = args.checkpoint or ROOT / settings["checkpoint"]
    train_path = args.train_file or ROOT / settings["train"]
    val_path = args.val_file or ROOT / settings["validation"]
    sources = {"checkpoint": checkpoint_path, "train": train_path, "validation": val_path}
    for name, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    check_output_dir(args, args.outdir, list(sources.values()))

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but is not available")

    # Select pairs deterministically; keep train and validation sets separate.
    x_train, y_train, train_meta = read_pairs(train_path, args.n_train, args.seed + 1)
    x_val, y_val, val_meta = read_pairs(val_path, args.n_val, args.seed + 2)
    model, original_checkpoint = load_hybrid_step_map_1d(str(checkpoint_path), device=device)
    dt, nx = validate_data(args.problem, original_checkpoint, train_meta, val_meta, x_train, x_val)
    parameters, original_coupling, active_count = configure_trainable(model, args.trainable)
    optimizer = make_optimizer(args, parameters)
    scheduler = make_scheduler(args, optimizer)
    loss_fn = get_loss_function(args.problem)
    output_dir = args.outdir.resolve()
    print(f"[setup] problem={args.problem} device={device} dtype={torch.get_default_dtype()} "
          f"nx={nx} dt={dt} train={len(x_train)} val={len(x_val)} "
          f"effective_trainable={active_count}")

    # Run either a short timing benchmark or the full training procedure.
    if args.mode == "benchmark":
        run_benchmark(args, model, optimizer, original_coupling, x_train, y_train,
                      loss_fn, dt, device, output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    config = experiment_config(args, sources)
    run_training(args, model, optimizer, scheduler, original_coupling, original_checkpoint,
                 x_train, y_train, x_val, y_val, loss_fn, dt, device, output_dir, config)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fold-pure NuCLR baseline with the official calcium objective and ~50k-step budget.

Protocol
--------
For each dataset x outer biological fold x seed:
  * use the pinned official NuCLR source tree under baselines/official/third_party;
  * random-initialize the official NuclrV2gaCa2 backbone + official projector;
  * train ONLY on OUTER-TRAIN worms with the official same-neuron two-view SSL;
  * use official calcium hyperparameters (30 s views, max distance 240 s,
    batch size 16, BF16, UnitDropout 0.5, AdamW, max LR 1.25e-4,
    one-epoch warmup, cosine decay, grad clip 1.0);
  * execute exactly target_train_steps optimizer updates (default 50,000),
    which instantiates the official README's "roughly 50,000 steps" guidance
    robustly for an NPZ sampler whose epoch length can vary by one batch;
  * evaluate OUTER-VALIDATION periodically only to choose a checkpoint;
  * open OUTER-TEST only after checkpoint selection is locked;
  * evaluate backbone embeddings with deterministic sequential 30 s windows,
    mean aggregation, L2-normalized cosine, and the shared benchmark evaluator.

Canonical cross-worm cell identity is NEVER used in the SSL loss.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
BENCH = ROOT / "baselines" / "official"
OFFICIAL_REPO = BENCH / "third_party" / "nuclr_official"
STAGE1_HELPER = ROOT / "archive" / "legacy_experiments" / "ey" / "stage1_pretrain_ey_nuclr_official_50k.py"
RUN_ROOT = BENCH / "runs" / "nuclr_official_scratch50k_cv5x3"

TARGET_TRAIN_STEPS = 50_000
VAL_EVERY_STEPS = 1_000
SAVE_LAST_EVERY_STEPS = 1_000
EVAL_WINDOW_STRIDE_SECONDS = 30.0
EVAL_WINDOW_BATCH_SIZE = 8

# -----------------------------------------------------------------------------
# Import local benchmark split/evaluator first, then force NuCLR imports to the
# pinned official source tree. This keeps benchmark plumbing local while model,
# loss, sampler, dataset, and utility modules are official.
# -----------------------------------------------------------------------------
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.lib.benchmark_cv5x3_common as common  # noqa: E402
from scripts.lib.fair_identity_protocol import select_geometry_medoid  # noqa: E402

# Fail loudly if importing the benchmark code already polluted the `src` package.
_preloaded_src = {
    k: getattr(v, "__file__", None)
    for k, v in sys.modules.items()
    if k == "src" or k.startswith("src.")
}
if _preloaded_src:
    raise RuntimeError(
        "benchmark_cv5x3_common imported a `src` package before the official "
        f"NuCLR source could be isolated: {_preloaded_src}"
    )
if not (OFFICIAL_REPO / "src").is_dir():
    raise FileNotFoundError(
        f"Missing official NuCLR checkout: {OFFICIAL_REPO}. "
        "Run setup_official_benchmark_repos.sh first."
    )
sys.path.insert(0, str(OFFICIAL_REPO))

# Load the existing NPZ adapter helper after official repo is first on sys.path.
if not STAGE1_HELPER.is_file():
    raise FileNotFoundError(STAGE1_HELPER)
spec = importlib.util.spec_from_file_location("nuclr_npz_adapter_helper", STAGE1_HELPER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import helper: {STAGE1_HELPER}")
s1 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = s1
spec.loader.exec_module(s1)

# Assert that helper imports resolved to the pinned official checkout.
def _assert_official_module(module: Any, name: str) -> None:
    p = Path(module.__file__).resolve()
    try:
        p.relative_to(OFFICIAL_REPO.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{name} resolved outside official NuCLR repo: {p}") from exc

import src.dataset as official_dataset  # noqa: E402
import src.losses.sw_contrastive as official_loss  # noqa: E402
import src.models.nuclr_v2ga_ca2 as official_model  # noqa: E402
import src.samplers as official_samplers  # noqa: E402
import src.utils as official_utils  # noqa: E402
for _mod, _name in (
    (official_dataset, "src.dataset"),
    (official_loss, "src.losses.sw_contrastive"),
    (official_model, "src.models.nuclr_v2ga_ca2"),
    (official_samplers, "src.samplers"),
    (official_utils, "src.utils"),
):
    _assert_official_module(_mod, _name)

# The pinned official source passes ``device=`` to xFormers' mask factory.
# Newer xFormers releases removed that keyword while retaining identical mask
# construction from sequence lengths.  Keep the official checkout pristine and
# bridge only this API drift in the local benchmark adapter.
_mask_cls = official_model.xops.fmha.BlockDiagonalMask
if "device" not in inspect.signature(_mask_cls.from_seqlens).parameters:
    _from_seqlens = _mask_cls.from_seqlens

    def _from_seqlens_compat(*args: Any, device: Any = None, **kwargs: Any):
        del device
        return _from_seqlens(*args, **kwargs)

    _mask_cls.from_seqlens = staticmethod(_from_seqlens_compat)

# Guard against drift in the helper's copied official calcium constants.
_EXPECTED = {
    "OFFICIAL_VIEW_SECONDS": 30.0,
    "OFFICIAL_MAX_VIEW_DISTANCE_SECONDS": 240.0,
    "OFFICIAL_BATCH_SIZE": 16,
    "OFFICIAL_PRECISION": "bf16",
    "OFFICIAL_MAX_LR": 1.25e-4,
    "OFFICIAL_LR_RISE_EPOCHS": 1,
    "OFFICIAL_LR_DECAY_START_EPOCH": 1,
    "OFFICIAL_WEIGHT_DECAY": 0.01,
    "OFFICIAL_GRAD_CLIP": 1.0,
    "OFFICIAL_UNIT_DROPOUT_MIN_FRACTION": 0.5,
    "OFFICIAL_LOSS_TAU": 0.2,
    "OFFICIAL_LOSS_DCL": True,
    "OFFICIAL_LOSS_PROJECTOR": True,
    "OFFICIAL_LOSS_FULL_DENOM": True,
}
for _key, _value in _EXPECTED.items():
    if getattr(s1, _key) != _value:
        raise RuntimeError(
            f"Local NPZ adapter constant {_key}={getattr(s1, _key)!r} "
            f"does not match pinned official setting {_value!r}"
        )


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_info(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, cwd=repo, text=True, capture_output=True, check=False)

    commit = run("git", "rev-parse", "HEAD")
    if commit.returncode != 0:
        raise RuntimeError(commit.stderr.strip() or "git rev-parse failed")
    status = run("git", "status", "--porcelain")
    branch = run("git", "symbolic-ref", "--short", "-q", "HEAD")
    return {
        "path": str(repo.resolve()),
        "commit": commit.stdout.strip(),
        "branch": branch.stdout.strip() if branch.returncode == 0 and branch.stdout.strip() else "DETACHED",
        "dirty": bool(status.stdout.strip()),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def locked_train_val_context(dataset: str, fold: int, seed: int):
    """Resolve train/val only. Outer test is deliberately not touched here."""
    _, b_dir = common.run_dirs(dataset, fold, seed)
    cfg = json.loads((b_dir / "config.json").read_text(encoding="utf-8"))
    locked_args = common.path_namespace(cfg, "cpu")
    train_paths = common.read_list(locked_args.train_list)
    val_paths = common.read_list(locked_args.val_list)
    label_to_int = common.legacy.base.collect_label_mapping(train_paths + val_paths)
    train_common = common.legacy.load_split(train_paths, label_to_int, locked_args)
    val_common = common.legacy.load_split(val_paths, label_to_int, locked_args)
    return cfg, locked_args, label_to_int, train_common, val_common


def make_nuclr_records(common_records: Sequence[Any], split_name: str) -> list[Any]:
    """Convert benchmark records to the official NPZ adapter with per-worm fs.

    Important boundary convention
    -----------------------------
    A regularly sampled trace with T samples at sampling rate fs has observed
    timestamps 0, 1/fs, ..., (T-1)/fs.  The legacy helper's WormRecord reports
    T/fs as ``duration_seconds`` (a half-open interval endpoint), which is fine
    for dense source traces but can let the official 30-s sampler place a window
    whose 128-point NuCLR interpolation grid extends beyond the *last observed
    timestamp* when source fs is low.

    For this benchmark we therefore expose the actual observed temporal support,
    (T-1)/fs, to both the official sampler and deterministic evaluator.  This
    does not clip/extrapolate any activity values and does not change the NuCLR
    model, objective, optimizer, or 50k-step budget; it only prevents incomplete
    end-of-recording windows from being sampled.
    """
    records = []
    for idx, rec in enumerate(common_records):
        activity, fs = common.load_activity(rec)
        path = common.source_npz(rec).resolve()
        traces = s1.normalize_traces(activity, "zscore")
        if traces.shape[0] != len(rec.labels):
            raise RuntimeError(
                f"Neuron count mismatch for {rec.worm_id}: activity={traces.shape[0]} labels={len(rec.labels)}"
            )

        fs = float(fs)
        num_timepoints = int(traces.shape[1])
        if num_timepoints < 2:
            raise RuntimeError(f"{rec.worm_id}: need at least 2 activity samples")

        observed_last_time = float((num_timepoints - 1) / fs)
        nominal_half_open_duration = float(num_timepoints / fs)
        if observed_last_time < 2.0 * s1.OFFICIAL_VIEW_SECONDS:
            raise RuntimeError(
                f"{rec.worm_id}: observed support={observed_last_time:.3f}s < 60s "
                "required by official TwoViewDataset semantics"
            )

        # EYOfficialDataAdapter is structurally typed at runtime: it only needs
        # these attributes/properties.  Use the *observed* support instead of
        # s1.WormRecord.duration_seconds (= T/fs) to make interpolation bounds
        # exact for low-sampling-rate recordings.
        records.append(
            SimpleNamespace(
                recording_id=f"{split_name}:{idx:04d}:{rec.worm_id}",
                path=path,
                traces=traces,
                source_fs=fs,
                num_neurons=int(traces.shape[0]),
                num_timepoints=num_timepoints,
                duration_seconds=observed_last_time,
                nominal_half_open_duration_seconds=nominal_half_open_duration,
            )
        )
    return records


def official_lr_for_exact_budget(
    global_step: int,
    steps_per_epoch: int,
    target_steps: int,
) -> float:
    """Official 1-epoch linear rise + cosine decay, expressed in step space.

    The official code schedules against steps_per_epoch * num_epochs. Here the
    NPZ RandomFixedWindowSampler can change full-batch count by one across epochs,
    so we schedule to the requested ~50k budget directly and stop at exactly that
    many optimizer updates. This prevents a post-cosine LR rebound while retaining
    the official schedule shape and max LR.
    """
    warmup_steps = max(int(steps_per_epoch), 1)
    max_lr = float(s1.OFFICIAL_MAX_LR)
    if global_step < warmup_steps:
        return max_lr * float(global_step) / float(warmup_steps)
    denom = max(int(target_steps) - warmup_steps, 1)
    progress = min(max((global_step - warmup_steps) / denom, 0.0), 1.0)
    return max_lr * (1.0 + math.cos(math.pi * progress)) * 0.5


def _window_starts(duration_seconds: float, stride_seconds: float) -> list[float]:
    max_start = duration_seconds - float(s1.OFFICIAL_VIEW_SECONDS)
    if max_start < -1e-8:
        return []
    starts = []
    x = 0.0
    while x <= max_start + 1e-8:
        starts.append(float(x))
        x += stride_seconds
    if not starts:
        starts = [0.0]
    return starts


@torch.inference_mode()
def embedding_map(
    model: torch.nn.Module,
    nuclr_records: Sequence[Any],
    common_records: Sequence[Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    if len(nuclr_records) != len(common_records):
        raise RuntimeError("NuCLR/common record count mismatch")

    model.eval()
    output: dict[str, np.ndarray] = {}
    for nrec, crec in zip(nuclr_records, common_records):
        n = int(nrec.num_neurons)
        if n != len(crec.labels):
            raise RuntimeError(f"Neuron count mismatch for {crec.worm_id}")

        adapter = s1.EYOfficialDataAdapter(
            records=[nrec],
            model_num_samples=int(model.num_latents * model.patch_size),
            seed=0,
        )
        starts = _window_starts(nrec.duration_seconds, EVAL_WINDOW_STRIDE_SECONDS)
        if not starts:
            raise RuntimeError(f"No valid evaluation windows for {crec.worm_id}")

        chunks = []
        for start_idx in range(0, len(starts), EVAL_WINDOW_BATCH_SIZE):
            sub = starts[start_idx:start_idx + EVAL_WINDOW_BATCH_SIZE]
            bins = []
            for start in sub:
                index = SimpleNamespace(
                    recording_id=nrec.recording_id,
                    start=float(start),
                    end=float(start + s1.OFFICIAL_VIEW_SECONDS),
                )
                full_view = adapter.extract_official_grid(index)
                bins.append(s1.traces_to_bins(full_view, model, device))

            all_bins = torch.cat(bins, dim=0)
            unit_seqlen = torch.tensor([n] * len(sub), dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                z = model(bins=all_bins, unit_seqlen=unit_seqlen)
            z = z.float().reshape(len(sub), n, -1).cpu().numpy()
            chunks.append(z)

        all_windows = np.concatenate(chunks, axis=0)
        emb = np.asarray(all_windows.mean(axis=0), dtype=np.float64)
        if emb.shape[0] != n or not np.isfinite(emb).all():
            raise RuntimeError(f"Invalid embedding block for {crec.worm_id}: {emb.shape}")
        output[str(crec.worm_id)] = common.normalize_rows(emb)
    return output


def evaluate_model(
    model: torch.nn.Module,
    nuclr_records: Sequence[Any],
    common_records: Sequence[Any],
    device: torch.device,
    dataset: str,
    fold: int,
    seed: int,
    method: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    z = embedding_map(model, nuclr_records, common_records, device)

    def score_fn(a: Any, b: Any):
        return z[str(a.worm_id)] @ z[str(b.worm_id)].T

    q = common.evaluate_pairwise(
        common_records,
        score_fn,
        dataset=dataset,
        fold=fold,
        seed=seed,
        method=method,
    )
    if q.empty:
        raise RuntimeError("No eligible queries in evaluation")
    ranks = q["rank"].to_numpy(float)
    metrics = {
        "queries": int(len(q)),
        "ranking_top1": float(q["top1"].mean()),
        "top3": float(q["top3"].mean()),
        "top5": float(q["top5"].mean()),
        "top10": float(np.mean(ranks <= 10)),
        "mrr": float(q["rr"].mean()),
        "mean_rank": float(np.mean(ranks)),
        "assignment_top1": float(q["hungarian_top1"].mean()),
    }
    return metrics, q


def evaluate_model_against_template(
    model: torch.nn.Module,
    query_nuclr_records: Sequence[Any],
    query_common_records: Sequence[Any],
    template_nuclr_record: Any,
    template_common_record: Any,
    device: torch.device,
    dataset: str,
    fold: int,
    seed: int,
    method: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    z = embedding_map(
        model,
        list(query_nuclr_records) + [template_nuclr_record],
        list(query_common_records) + [template_common_record],
        device,
    )

    def score_fn(query: Any, reference: Any):
        return z[str(query.worm_id)] @ z[str(reference.worm_id)].T

    q = common.evaluate_against_fixed_reference(
        query_common_records,
        template_common_record,
        score_fn,
        dataset=dataset,
        fold=fold,
        seed=seed,
        method=method,
    )
    if q.empty:
        raise RuntimeError("No eligible fixed-template queries in evaluation")
    ranks = q["rank"].to_numpy(float)
    metrics = {
        "queries": int(len(q)),
        "ranking_top1": float(q["top1"].mean()),
        "top3": float(q["top3"].mean()),
        "top5": float(q["top5"].mean()),
        "top10": float(np.mean(ranks <= 10)),
        "mrr": float(q["rr"].mean()),
        "mean_rank": float(np.mean(ranks)),
        "assignment_top1": float(q["hungarian_top1"].mean()),
    }
    return metrics, q


def is_better(candidate: dict[str, float], best: dict[str, float] | None) -> bool:
    if best is None:
        return True
    if candidate["ranking_top1"] > best["ranking_top1"] + 1e-12:
        return True
    if abs(candidate["ranking_top1"] - best["ranking_top1"]) <= 1e-12:
        return candidate["mrr"] > best["mrr"] + 1e-12
    return False


def rng_state(adapter: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "second_view_rng": adapter.second_view_rng.bit_generator.state,
    }


def restore_rng(state: dict[str, Any], adapter: Any) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    adapter.second_view_rng.bit_generator.state = state["second_view_rng"]


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    loss_module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: str,
    fold: int,
    seed: int,
    epoch: int,
    next_batch_idx: int,
    global_step: int,
    steps_per_epoch: int,
    target_steps: int,
    train_loss: float | None,
    val_metrics: dict[str, float] | None,
    adapter: Any,
    official_repo: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "nuclr_official_scratch50k_cv5x3",
            "dataset": dataset,
            "fold": int(fold),
            "seed": int(seed),
            "epoch": int(epoch),
            "next_batch_idx": int(next_batch_idx),
            "global_step": int(global_step),
            "steps_per_epoch_setup": int(steps_per_epoch),
            "target_train_steps": int(target_steps),
            "train_loss": None if train_loss is None else float(train_loss),
            "validation": val_metrics,
            "model_state_dict": model.state_dict(),
            "loss_state_dict": loss_module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": rng_state(adapter),
            "official_repo": official_repo,
            "official_config": {
                "views.duration": float(s1.OFFICIAL_VIEW_SECONDS),
                "views.max_distance": float(s1.OFFICIAL_MAX_VIEW_DISTANCE_SECONDS),
                "batch_size": int(s1.OFFICIAL_BATCH_SIZE),
                "precision": str(s1.OFFICIAL_PRECISION),
                "lr.max": float(s1.OFFICIAL_MAX_LR),
                "lr.rise_epochs": int(s1.OFFICIAL_LR_RISE_EPOCHS),
                "lr.decay_start_epoch": int(s1.OFFICIAL_LR_DECAY_START_EPOCH),
                "grad_clip.max_norm": float(s1.OFFICIAL_GRAD_CLIP),
                "weight_decay": float(s1.OFFICIAL_WEIGHT_DECAY),
                "train_transform.min_units": float(s1.OFFICIAL_UNIT_DROPOUT_MIN_FRACTION),
                "loss.tau": float(s1.OFFICIAL_LOSS_TAU),
                "loss.dcl": bool(s1.OFFICIAL_LOSS_DCL),
                "loss.projector": bool(s1.OFFICIAL_LOSS_PROJECTOR),
                "loss.full_denom": bool(s1.OFFICIAL_LOSS_FULL_DENOM),
            },
        },
        path,
    )


def train_one(
    *,
    dataset: str,
    fold: int,
    seed: int,
    train_records: Sequence[Any],
    val_records: Sequence[Any],
    val_common: Sequence[Any],
    device: torch.device,
    out_dir: Path,
    target_steps: int,
    val_every_steps: int,
    save_last_every_steps: int,
    official_repo: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    seed_everything(seed)
    model = s1.build_official_model(device)
    loss_module = s1.build_official_loss(model, device)
    optimizer = s1.build_official_optimizer(model, loss_module)

    adapter = s1.EYOfficialDataAdapter(
        records=train_records,
        model_num_samples=int(model.num_latents * model.patch_size),
        seed=seed,
    )
    sampler = adapter.make_first_view_sampler(seed=seed)
    sampler.set_epoch(0)
    epoch0_indices = list(iter(sampler))
    steps_per_epoch = len(epoch0_indices) // int(s1.OFFICIAL_BATCH_SIZE)
    if steps_per_epoch < 1:
        raise RuntimeError(f"Too few official SSL samples: {len(epoch0_indices)}")

    history_path = out_dir / "validation_history.csv"
    best_path = out_dir / "selected" / "best.pt"
    last_path = out_dir / "checkpoints" / "last.pt"
    history: list[dict[str, Any]] = []
    best_metrics: dict[str, float] | None = None
    best_step = -1

    epoch = 0
    next_batch_idx = 0
    global_step = 0

    # Resume is intentionally from our own scratch-50k checkpoint only.
    if resume and last_path.is_file():
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        if ckpt.get("stage") != "nuclr_official_scratch50k_cv5x3":
            raise RuntimeError(f"Refusing incompatible resume checkpoint: {last_path}")
        if int(ckpt.get("target_train_steps", -1)) != int(target_steps):
            raise RuntimeError("Resume target_train_steps mismatch")
        if int(ckpt.get("steps_per_epoch_setup", -1)) != int(steps_per_epoch):
            raise RuntimeError("Resume steps_per_epoch mismatch")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        loss_module.load_state_dict(ckpt["loss_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        epoch = int(ckpt["epoch"])
        next_batch_idx = int(ckpt["next_batch_idx"])
        global_step = int(ckpt["global_step"])
        restore_rng(ckpt["rng_state"], adapter)
        print(f"[RESUME] step={global_step} epoch={epoch} next_batch={next_batch_idx}", flush=True)

    if history_path.is_file():
        try:
            old = pd.read_csv(history_path)
            history = old.to_dict("records")
            eligible = old[old["eligible_for_selection"] == True]  # noqa: E712
            if len(eligible):
                eligible = eligible.sort_values(["ranking_top1", "mrr", "global_step"], ascending=[False, False, True])
                row = eligible.iloc[0]
                best_metrics = {
                    "queries": int(row["queries"]),
                    "ranking_top1": float(row["ranking_top1"]),
                    "top3": float(row["top3"]),
                    "top5": float(row["top5"]),
                    "top10": float(row["top10"]),
                    "mrr": float(row["mrr"]),
                    "mean_rank": float(row["mean_rank"]),
                    "assignment_top1": float(row["assignment_top1"]),
                }
                best_step = int(row["global_step"])
        except Exception:
            history = []
            best_metrics = None
            best_step = -1

    # Random-init validation is diagnostic only and can never be selected.
    if global_step == 0 and not history:
        init_metrics, _ = evaluate_model(
            model, val_records, val_common, device, dataset, fold, seed,
            method="NuCLR official scratch50k val",
        )
        history.append({
            "epoch": 0,
            "global_step": 0,
            "lr": 0.0,
            "train_loss": np.nan,
            "eligible_for_selection": False,
            **init_metrics,
        })
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(
            f"[{dataset} f{fold} s{seed}] random-init val Top1={init_metrics['ranking_top1']:.4f} "
            f"Top5={init_metrics['top5']:.4f} MRR={init_metrics['mrr']:.4f}",
            flush=True,
        )

    next_val_step = ((global_step // val_every_steps) + 1) * val_every_steps
    next_save_step = ((global_step // save_last_every_steps) + 1) * save_last_every_steps

    running_loss = 0.0
    running_matches = 0
    while global_step < target_steps:
        model.train()
        loss_module.train()
        sampler.set_epoch(epoch)
        first_indices = list(iter(sampler))
        batches = list(s1.iter_full_batches(first_indices, int(s1.OFFICIAL_BATCH_SIZE)))
        if not batches:
            raise RuntimeError(f"Epoch {epoch}: no full official batches")
        if next_batch_idx > len(batches):
            raise RuntimeError(
                f"Resume next_batch_idx={next_batch_idx} exceeds epoch batches={len(batches)}"
            )

        for batch_idx, first_batch in enumerate(batches):
            if batch_idx < next_batch_idx:
                continue
            if global_step >= target_steps:
                break

            lr = official_lr_for_exact_budget(global_step, steps_per_epoch, target_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            prepared = s1.prepare_batch(
                first_indices=first_batch,
                adapter=adapter,
                model=model,
                device=device,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                e1 = model(bins=prepared.bins1, unit_seqlen=prepared.unit_seqlen1)
                e2 = model(bins=prepared.bins2, unit_seqlen=prepared.unit_seqlen2)
                loss = loss_module(
                    e1, e2, prepared.metadata,
                    prefix="train", logger=None, return_loss_dict=False,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss dataset={dataset} fold={fold} seed={seed} step={global_step}"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(s1.OFFICIAL_GRAD_CLIP))
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), float(s1.OFFICIAL_GRAD_CLIP))
            optimizer.step()
            s1.validate_weights(model)
            s1.validate_weights(loss_module)

            running_loss += float(loss.detach().item()) * int(prepared.num_matches)
            running_matches += int(prepared.num_matches)
            global_step += 1
            next_batch_idx = batch_idx + 1

            should_validate = global_step >= next_val_step or global_step >= target_steps
            if should_validate:
                train_loss = running_loss / max(running_matches, 1)
                val_metrics, _ = evaluate_model(
                    model, val_records, val_common, device, dataset, fold, seed,
                    method="NuCLR official scratch50k val",
                )
                row = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "lr": lr,
                    "train_loss": train_loss,
                    "eligible_for_selection": True,
                    **val_metrics,
                }
                history.append(row)
                pd.DataFrame(history).to_csv(history_path, index=False)
                print(
                    f"[{dataset} f{fold} s{seed}] step={global_step:05d}/{target_steps:05d} "
                    f"epoch={epoch:04d} loss={train_loss:.6f} lr={lr:.3e} "
                    f"val Top1={val_metrics['ranking_top1']:.4f} "
                    f"Top5={val_metrics['top5']:.4f} MRR={val_metrics['mrr']:.4f}",
                    flush=True,
                )
                if is_better(val_metrics, best_metrics):
                    best_metrics = dict(val_metrics)
                    best_step = int(global_step)
                    save_checkpoint(
                        best_path,
                        model=model,
                        loss_module=loss_module,
                        optimizer=optimizer,
                        dataset=dataset,
                        fold=fold,
                        seed=seed,
                        epoch=epoch,
                        next_batch_idx=next_batch_idx,
                        global_step=global_step,
                        steps_per_epoch=steps_per_epoch,
                        target_steps=target_steps,
                        train_loss=train_loss,
                        val_metrics=val_metrics,
                        adapter=adapter,
                        official_repo=official_repo,
                    )
                while next_val_step <= global_step:
                    next_val_step += val_every_steps
                running_loss = 0.0
                running_matches = 0
                model.train()
                loss_module.train()

            if global_step >= next_save_step or global_step >= target_steps:
                save_checkpoint(
                    last_path,
                    model=model,
                    loss_module=loss_module,
                    optimizer=optimizer,
                    dataset=dataset,
                    fold=fold,
                    seed=seed,
                    epoch=epoch,
                    next_batch_idx=next_batch_idx,
                    global_step=global_step,
                    steps_per_epoch=steps_per_epoch,
                    target_steps=target_steps,
                    train_loss=None,
                    val_metrics=None,
                    adapter=adapter,
                    official_repo=official_repo,
                )
                while next_save_step <= global_step:
                    next_save_step += save_last_every_steps

        # Move to next deterministic sampler epoch.
        if next_batch_idx >= len(batches):
            epoch += 1
            next_batch_idx = 0

    if best_metrics is None or not best_path.is_file():
        raise RuntimeError("No trained validation checkpoint was selected")

    return {
        "best_checkpoint": str(best_path.resolve()),
        "best_global_step": int(best_step),
        "best_validation": best_metrics,
        "target_train_steps": int(target_steps),
        "final_global_step": int(global_step),
        "setup_steps_per_epoch": int(steps_per_epoch),
        "epochs_touched": int(epoch + (1 if next_batch_idx > 0 else 0)),
    }


def run_one(
    dataset: str,
    fold: int,
    seed: int,
    device_name: str,
    run_root: Path,
    target_steps: int,
    val_every_steps: int,
    save_last_every_steps: int,
    resume: bool,
) -> None:
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Official NuCLR calcium baseline requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Official NuCLR calcium config requires BF16-capable CUDA")
    if target_steps <= 0 or val_every_steps <= 0 or save_last_every_steps <= 0:
        raise ValueError("step counts must be positive")

    repo = git_info(OFFICIAL_REPO)
    if repo["dirty"]:
        raise RuntimeError(
            f"Official NuCLR checkout is dirty; refusing benchmark run: {OFFICIAL_REPO}"
        )

    out_dir = run_root / dataset / f"fold_{fold}" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "outer_test_medoid_template_v1" / "result.json"
    if result_path.is_file():
        print("[REUSE COMPLETE]", result_path)
        return

    cfg, locked_args, label_to_int, train_common, val_common = locked_train_val_context(
        dataset, fold, seed
    )
    train_records = make_nuclr_records(train_common, "train")
    val_records = make_nuclr_records(val_common, "val")
    train_geometry = []
    for record in train_common:
        xyz = record.xyz.detach().cpu().numpy() if torch.is_tensor(record.xyz) else np.asarray(record.xyz)
        xyz = np.asarray(xyz, dtype=np.float64)[:, :3]
        train_geometry.append((xyz - np.median(xyz, axis=0, keepdims=True)) / 200.0)
    medoid = select_geometry_medoid(
        [str(record.worm_id) for record in train_common], train_geometry
    )
    template_common = train_common[medoid.index]
    template_record = train_records[medoid.index]

    # Setup sampler once to record the fold-specific official batch budget.
    tmp_adapter = s1.EYOfficialDataAdapter(train_records, model_num_samples=128, seed=seed)
    tmp_sampler = tmp_adapter.make_first_view_sampler(seed=seed)
    tmp_sampler.set_epoch(0)
    setup_samples = len(list(iter(tmp_sampler)))
    setup_steps = setup_samples // int(s1.OFFICIAL_BATCH_SIZE)
    if setup_steps < 1:
        raise RuntimeError("No full official batches in outer train split")

    protocol = {
        "dataset": dataset,
        "fold": int(fold),
        "seed": int(seed),
        "method": "NuCLR (official scratch, 50k SSL)",
        "role": "activity-only baseline",
        "official_repo": repo,
        "outer_train_worms": [str(x.worm_id) for x in train_common],
        "outer_val_worms": [str(x.worm_id) for x in val_common],
        "outer_test_opened_before_selection": False,
        "template_selection": {
            "selection_split": "outer_train_only",
            "uses_validation": False,
            "uses_test": False,
            "uses_identity_labels": False,
            "distance": "symmetric mean nearest-neighbour Euclidean distance after per-animal median centering and /200 scaling",
            "criterion": "minimum mean distance to all other outer-training worms",
            "tie_break": "worm UID, then train-list index",
            "template_worm": str(template_common.worm_id),
            "template_train_index": int(medoid.index),
            "template_mean_distance": medoid.mean_distance,
            "training_candidates": [
                {"worm_id": str(record.worm_id), "mean_geometry_distance": medoid.mean_distances[index]}
                for index, record in enumerate(train_common)
            ],
        },
        "training": {
            "initialization": "random per seed; no EY or external checkpoint",
            "architecture": "official NuclrV2gaCa2",
            "objective": "official SampleWiseContrastiveLoss (same local neuron, two temporal views, same worm)",
            "cross_worm_identity_supervision": False,
            "views_seconds": float(s1.OFFICIAL_VIEW_SECONDS),
            "max_view_distance_seconds": float(s1.OFFICIAL_MAX_VIEW_DISTANCE_SECONDS),
            "unit_dropout_min_fraction": float(s1.OFFICIAL_UNIT_DROPOUT_MIN_FRACTION),
            "batch_size": int(s1.OFFICIAL_BATCH_SIZE),
            "precision": str(s1.OFFICIAL_PRECISION),
            "optimizer": "AdamW, official decay/no-decay grouping",
            "weight_decay": float(s1.OFFICIAL_WEIGHT_DECAY),
            "max_lr": float(s1.OFFICIAL_MAX_LR),
            "lr_schedule": "official one-epoch linear rise + cosine decay, instantiated in exact optimizer-step space",
            "grad_clip": float(s1.OFFICIAL_GRAD_CLIP),
            "target_optimizer_steps": int(target_steps),
            "setup_sampler_samples": int(setup_samples),
            "setup_steps_per_epoch": int(setup_steps),
            "rough_equivalent_epochs": float(target_steps / setup_steps),
            "no_ssl_early_stopping": True,
        },
        "selection": {
            "split": "outer validation only",
            "metric": "Direct Top-1",
            "tie_break": "MRR",
            "validation_every_optimizer_steps": int(val_every_steps),
            "random_init_not_eligible": True,
        },
        "evaluation": {
            "space": "backbone",
            "window_seconds": float(s1.OFFICIAL_VIEW_SECONDS),
            "window_stride_seconds": float(EVAL_WINDOW_STRIDE_SECONDS),
            "aggregation": "mean across deterministic sequential windows",
            "score": "L2-normalized cosine",
            "candidate_universe": "all reference neurons",
            "shared_evaluator": "benchmark_cv5x3_common.evaluate_against_fixed_reference",
            "reference_policy": "single outer-training geometry medoid",
        },
    }
    json_dump(out_dir / "protocol.json", protocol)

    selected = train_one(
        dataset=dataset,
        fold=fold,
        seed=seed,
        train_records=train_records,
        val_records=val_records,
        val_common=val_common,
        device=device,
        out_dir=out_dir,
        target_steps=target_steps,
        val_every_steps=val_every_steps,
        save_last_every_steps=save_last_every_steps,
        official_repo=repo,
        resume=resume,
    )
    json_dump(out_dir / "selected" / "selection.json", selected)
    print("\nSELECTED")
    print(json.dumps(selected, indent=2), flush=True)

    # ------------------------------------------------------------------
    # OUTER TEST IS FIRST READ HERE, AFTER THE 50k TRAINING TRAJECTORY AND
    # VALIDATION-ONLY CHECKPOINT SELECTION ARE COMPLETE.
    # ------------------------------------------------------------------
    test_paths = common.read_list(locked_args.test_list)
    test_common = common.legacy.load_split(test_paths, label_to_int, locked_args)
    test_records = make_nuclr_records(test_common, "test")

    # Rebuild random architecture, then load the selected official checkpoint.
    model = s1.build_official_model(device)
    ckpt = torch.load(selected["best_checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    test_metrics, query_rows = evaluate_model_against_template(
        model,
        test_records,
        test_common,
        template_record,
        template_common,
        device,
        dataset,
        fold,
        seed,
        method="NuCLR (official scratch, 50k SSL)",
    )

    outer = out_dir / "outer_test_medoid_template_v1"
    outer.mkdir(parents=True, exist_ok=True)
    query_rows.to_csv(outer / "query_level.csv", index=False)
    result = {
        "dataset": dataset,
        "fold": int(fold),
        "seed": int(seed),
        "method": "NuCLR (official scratch, 50k SSL)",
        "official_repo": repo,
        "selected_global_step": int(selected["best_global_step"]),
        "selected_checkpoint": selected["best_checkpoint"],
        "target_train_steps": int(target_steps),
        "final_global_step": int(selected["final_global_step"]),
        "metrics": test_metrics,
        "protocol": {
            "random_initialization": True,
            "external_pretraining": False,
            "train_loss_uses_cross_worm_cell_id": False,
            "test_opened_after_selection": True,
            "backbone_space": True,
            "cosine": True,
            "test_test_pairing": False,
            "reference_worm": str(template_common.worm_id),
        },
    }
    json_dump(outer / "result.json", result)
    print("\nOUTER TEST")
    print(json.dumps(test_metrics, indent=2), flush=True)



def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["atanas", "rld"], required=True)
    p.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    p.add_argument("--seed", type=int, choices=[1, 42, 123], required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--run-root", type=Path, default=RUN_ROOT)
    p.add_argument("--target-train-steps", type=int, default=TARGET_TRAIN_STEPS)
    p.add_argument("--val-every-steps", type=int, default=VAL_EVERY_STEPS)
    p.add_argument("--save-last-every-steps", type=int, default=SAVE_LAST_EVERY_STEPS)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    run_one(
        args.dataset,
        args.fold,
        args.seed,
        args.device,
        args.run_root,
        args.target_train_steps,
        args.val_every_steps,
        args.save_last_every_steps,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

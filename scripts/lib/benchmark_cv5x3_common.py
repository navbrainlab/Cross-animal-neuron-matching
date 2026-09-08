#!/usr/bin/env python3
from __future__ import annotations

import json, math, random, sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SNAPSHOT_CODE = ROOT / "paper_submission_data/5fold_cross_validation/code"
if str(SNAPSHOT_CODE) not in sys.path:
    sys.path.insert(0, str(SNAPSHOT_CODE))

import train_ambiguity_aware_geo_activity_transformer as candidate_b_trainer

legacy = candidate_b_trainer.legacy

FOLDS = (1, 2, 3, 4, 5)
SEEDS = (1, 42, 123)
RUN_ROOT = ROOT / 'runs' / 'fold_pure_table1init_candidates_ab_20260817'
DEFAULT_OUT = ROOT / 'paper_submission_data' / 'benchmark_cv5x3_20260818'

METHODS = (
    'CPD',
    'fDNC',
    'Statistical Atlas',
    'NeurPIR',
    'NuCLR',
    'fDNC + NuCLR Score Fusion',
    'MulT (adapted)',
    'Ours',
)


def run_dirs(dataset: str, fold: int, seed: int) -> tuple[Path, Path]:
    """Return the locked Candidate-A/B directories for one outer run."""
    pipeline = (
        RUN_ROOT / "folds" / dataset / f"fold_{fold}"
        / "pipeline" / f"seed_{seed}"
    )
    return pipeline / "candidate_a", pipeline / "candidate_b"


def path_namespace(cfg: dict[str, Any], device: str) -> Namespace:
    """Recreate the trainer namespace stored in a locked run config."""
    values = dict(cfg["args"])
    for key in (
        "train_list", "val_list", "test_list", "save_dir",
        "source_data_root", "external_test_list",
    ):
        value = values.get(key)
        if value not in (None, ""):
            values[key] = Path(value)
    values["device"] = device
    return Namespace(**values)


def all_pairs(n: int):
    if hasattr(legacy, "base") and hasattr(legacy.base, "all_unordered_pairs"):
        return legacy.base.all_unordered_pairs(n)
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_list(path: str | Path) -> list[str]:
    p = Path(path)
    rows = []
    for line in p.read_text(encoding='utf-8').splitlines():
        x = line.strip()
        if not x or x.startswith('#'):
            continue
        q = Path(x)
        rows.append(str((p.parent / q).resolve() if not q.is_absolute() else q.resolve()))
    return rows


def split_bundle(dataset: str, fold: int, seed: int, device: str = 'cpu'):
    '''Exact train/val/test lists used by the locked Candidate-A/B experiment.'''
    _, b_dir = run_dirs(dataset, fold, seed)
    cfg = json.loads((b_dir / 'config.json').read_text(encoding='utf-8'))
    args = path_namespace(cfg, device)
    train_paths, val_paths, test_paths = map(
        read_list, (args.train_list, args.val_list, args.test_list)
    )
    label_to_int = legacy.base.collect_label_mapping(train_paths + val_paths)
    return {
        'cfg': cfg,
        'args': args,
        'train_paths': train_paths,
        'val_paths': val_paths,
        'test_paths': test_paths,
        'label_to_int': label_to_int,
        'train': legacy.load_split(train_paths, label_to_int, args),
        'val': legacy.load_split(val_paths, label_to_int, args),
        'test': legacy.load_split(test_paths, label_to_int, args),
    }


def unique_label_map(labels: torch.Tensor) -> dict[int, int]:
    positions: dict[int, list[int]] = {}
    for index, value in enumerate(labels.detach().cpu().tolist()):
        label = int(value)
        if label >= 0:
            positions.setdefault(label, []).append(index)
    return {
        label: indices[0]
        for label, indices in positions.items()
        if len(indices) == 1
    }


def rank_of_gt(row: np.ndarray, gt: int) -> int:
    return 1 + int(np.sum(row > row[gt]))


def evaluate_score_matrix(
    score: np.ndarray,
    qrec: Any,
    rrec: Any,
    *,
    dataset: str,
    fold: int,
    seed: int,
    method: str,
) -> list[dict[str, Any]]:
    score = np.asarray(score, dtype=np.float64)
    if score.shape != (len(qrec.labels), len(rrec.labels)):
        raise ValueError(
            f'{method}: score={score.shape}, expected={(len(qrec.labels), len(rrec.labels))}'
        )
    if not np.isfinite(score).all():
        raise ValueError(f'{method}: score contains NaN/Inf')

    qmap, rmap = unique_label_map(qrec.labels), unique_label_map(rrec.labels)
    common = sorted(set(qmap) & set(rmap))
    if not common:
        return []

    rr, cc = linear_sum_assignment(-score)
    assignment = {int(r): int(c) for r, c in zip(rr, cc)}

    rows = []
    for lab in common:
        qi, gi = qmap[lab], rmap[lab]
        rank = rank_of_gt(score[qi], gi)
        rows.append({
            'dataset': dataset,
            'fold': int(fold),
            'seed': int(seed),
            'method': method,
            'query_worm': str(qrec.worm_id),
            'reference_worm': str(rrec.worm_id),
            'query_row': int(qi),
            'gt_label_int': int(lab),
            'rank': int(rank),
            'top1': float(rank == 1),
            'top3': float(rank <= 3),
            'top5': float(rank <= 5),
            'rr': 1.0 / float(rank),
            'hungarian_top1': float(assignment.get(qi, -1) == gi),
        })
    return rows


def evaluate_pairwise(
    records: Sequence[Any],
    score_fn,
    *,
    dataset: str,
    fold: int,
    seed: int,
    method: str,
) -> pd.DataFrame:
    rows = []
    for ia, ib in all_pairs(len(records)):
        a, b = records[ia], records[ib]
        s = np.asarray(score_fn(a, b), dtype=np.float64)
        rows.extend(evaluate_score_matrix(
            s, a, b, dataset=dataset, fold=fold, seed=seed, method=method
        ))
        rows.extend(evaluate_score_matrix(
            s.T, b, a, dataset=dataset, fold=fold, seed=seed, method=method
        ))
    return pd.DataFrame(rows)


def evaluate_against_fixed_reference(
    queries: Sequence[Any],
    reference: Any,
    score_fn,
    *,
    dataset: str,
    fold: int,
    seed: int,
    method: str,
) -> pd.DataFrame:
    """Evaluate held-out animals only against one outer-training template."""
    rows = []
    reference_uid = str(reference.worm_id)
    for query in queries:
        if str(query.worm_id) == reference_uid:
            raise ValueError("Query/reference overlap in fixed-template evaluation")
        score = np.asarray(score_fn(query, reference), dtype=np.float64)
        rows.extend(
            evaluate_score_matrix(
                score,
                query,
                reference,
                dataset=dataset,
                fold=fold,
                seed=seed,
                method=method,
            )
        )
    return pd.DataFrame(rows)


def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def zscore_matrix(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    return (s - s.mean()) / max(float(s.std()), 1e-8)


def nuclr_score(a, b) -> np.ndarray:
    za = normalize_rows(a.nuclr_emb.detach().cpu().numpy())
    zb = normalize_rows(b.nuclr_emb.detach().cpu().numpy())
    return za @ zb.T


def source_npz(record: Any) -> Path:
    candidates = []
    for attr in ('source_path', 'path', 'embedding_path'):
        value = getattr(record, attr, '')
        if value:
            candidates.append(Path(str(value)))
    for p in candidates:
        if not p.is_file():
            continue
        try:
            with np.load(p, allow_pickle=True) as d:
                if 'activity_raw' in d.files:
                    return p
                if 'source_path' in d.files:
                    raw = np.asarray(d['source_path']).reshape(-1)[0]
                    if isinstance(raw, bytes):
                        raw = raw.decode('utf-8')
                    q = Path(str(raw))
                    if q.is_file():
                        return q
        except Exception:
            pass
    raise FileNotFoundError(
        f'Could not resolve raw activity NPZ for worm={getattr(record, "worm_id", "?")}'
    )


_ACTIVITY_CACHE: dict[str, tuple[np.ndarray, float]] = {}


def load_activity(record: Any) -> tuple[np.ndarray, float]:
    cache_key = str(getattr(record, 'worm_id', '')) + '||' + str(getattr(record, 'source_path', getattr(record, 'path', '')))
    if cache_key in _ACTIVITY_CACHE:
        return _ACTIVITY_CACHE[cache_key]
    p = source_npz(record)
    with np.load(p, allow_pickle=True) as d:
        x = np.asarray(d['activity_raw'], dtype=np.float32)
        n = len(record.labels)
        if x.ndim != 2:
            raise ValueError(f'{p}: activity_raw must be 2-D, got {x.shape}')
        if x.shape[0] != n and x.shape[1] == n:
            x = x.T
        if x.shape[0] != n:
            raise ValueError(f'{p}: activity rows {x.shape[0]} != neurons {n}')
        fs = 4.0
        for key in ('sampling_rate_hz', 'source_fs', 'fs'):
            if key in d.files:
                value = float(np.asarray(d[key]).reshape(-1)[0])
                if np.isfinite(value) and value > 0:
                    fs = value
                    break
    _ACTIVITY_CACHE[cache_key] = (x, fs)
    return x, fs


def collapse_repeats(df: pd.DataFrame) -> pd.DataFrame:
    key = [
        'dataset', 'fold', 'method', 'query_worm', 'reference_worm',
        'query_row', 'gt_label_int',
    ]
    agg = {
        'top1': ('top1', 'mean'),
        'rr': ('rr', 'mean'),
        'hungarian_top1': ('hungarian_top1', 'mean'),
    }
    if df['top3'].notna().any():
        agg['top3'] = ('top3', 'mean')
    if df['top5'].notna().any():
        agg['top5'] = ('top5', 'mean')
    out = df.groupby(key, as_index=False).agg(**agg)
    if 'top3' not in out:
        out['top3'] = np.nan
    if 'top5' not in out:
        out['top5'] = np.nan
    return out


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method), g in df.groupby(['dataset', 'method'], sort=False):
        c = collapse_repeats(g)
        per_run = (
            g.groupby(['fold', 'seed'], as_index=False)
             .agg(top1=('top1', 'mean'), mrr=('rr', 'mean'))
        )
        rows.append({
            'dataset': dataset,
            'method': method,
            'queries': len(c),
            'top1': float(c.top1.mean()),
            'top3': float(c.top3.mean()) if c.top3.notna().any() else math.nan,
            'top5': float(c.top5.mean()) if c.top5.notna().any() else math.nan,
            'mrr': float(c.rr.mean()),
            'hungarian_top1': float(c.hungarian_top1.mean()) if c.hungarian_top1.notna().any() else math.nan,
            'run_top1_mean': float(per_run.top1.mean()),
            'run_top1_sd': float(per_run.top1.std(ddof=1)) if len(per_run) > 1 else 0.0,
            'run_mrr_mean': float(per_run.mrr.mean()),
            'run_mrr_sd': float(per_run.mrr.std(ddof=1)) if len(per_run) > 1 else 0.0,
            'folds': int(g.fold.nunique()),
            'seeds': int(g.seed.nunique()),
        })
    order = {m: i for i, m in enumerate(METHODS)}
    out = pd.DataFrame(rows)
    if len(out):
        out['_order'] = out.method.map(order).fillna(999)
        out = out.sort_values(['dataset', '_order']).drop(columns='_order')
    return out


def save_run(df: pd.DataFrame, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / 'query_level.csv', index=False)
    summary = {
        'queries': int(len(df)),
        'top1': float(df.top1.mean()) if len(df) else math.nan,
        'top3': float(df.top3.mean()) if len(df) else math.nan,
        'top5': float(df.top5.mean()) if len(df) else math.nan,
        'mrr': float(df.rr.mean()) if len(df) else math.nan,
        'hungarian_top1': float(df.hungarian_top1.mean()) if len(df) else math.nan,
    }
    (run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')

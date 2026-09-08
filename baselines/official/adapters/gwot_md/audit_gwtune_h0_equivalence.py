#!/usr/bin/env python3
"""Compare the Atanas h=0 implementation with the authors' GWTune solver.

GWTune has no multi-distance API, so this audit intentionally covers only the
single-relation h=0 boundary.  It imports the pinned checkout without changing
it.  A minimal Optuna stub is sufficient because the low-level solver under
test does not use study management.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
PAPER_DIR = ROOT / "baselines/gwot_md"
DEFAULT_GWTUNE = ROOT / "baselines" / "official" / "third_party" / "gwtune_official"
DEFAULT_POT = ROOT / "baselines" / "official" / "envs" / "gwtune_pot094"


def install_optuna_import_stub() -> None:
    """Provide only names needed while importing GWTune's low-level class."""
    if importlib.util.find_spec("optuna") is not None:
        return
    module = types.ModuleType("optuna")
    module.TrialPruned = type("TrialPruned", (Exception,), {})
    module.trial = types.SimpleNamespace(Trial=object)
    sys.modules["optuna"] = module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PAPER_DIR / "data_paper_snapshot_v1")
    parser.add_argument("--gwtune-root", type=Path, default=DEFAULT_GWTUNE)
    parser.add_argument("--pot-root", type=Path, default=DEFAULT_POT)
    parser.add_argument("--source", default="2022-06-14-01")
    parser.add_argument("--target", default="2022-06-14-07")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(PAPER_DIR))
    sys.path.insert(0, str(args.gwtune_root))
    if args.pot_root.is_dir():
        sys.path.insert(0, str(args.pot_root))
    install_optuna_import_stub()

    from solve_gwot_md_atanas21 import (  # pylint: disable=import-error,import-outside-toplevel
        delayed_cosine_relations,
        gw_objective,
        highpass,
        random_feasible_plan,
        solve_one_start,
    )
    from src.gw_alignment import (  # pylint: disable=import-error,import-outside-toplevel
        MainGromovWasserstainComputation,
    )
    from src.utils.init_matrix import InitMatrix  # pylint: disable=import-error,import-outside-toplevel
    import ot  # pylint: disable=import-outside-toplevel

    def relation(uid: str) -> np.ndarray:
        with np.load(args.data_dir / f"{uid}.npz", allow_pickle=False) as item:
            activity = np.asarray(item["activity_raw"], dtype=np.float64)
            sample_rate = float(item["sampling_rate_hz"])
        return delayed_cosine_relations(highpass(activity, sample_rate), 0)[0]

    source = relation(args.source)
    target = relation(args.target)
    m, n = source.shape[0], target.shape[0]
    official_initial = InitMatrix(m, n).make_initial_T("random", args.seed)
    local_initial = random_feasible_plan(m, n, args.seed)

    official_solver = MainGromovWasserstainComputation(
        source,
        target,
        device="cpu",
        to_types="numpy",
        data_type="double",
        max_iter=1000,
        numItermax=1000,
        n_iter=1,
        fix_random_init_seed=1,
        gw_type="entropic_gromov_wasserstein",
        sinkhorn_method="sinkhorn_log",
        tol=1e-9,
    )
    official = official_solver.gw_computation(args.epsilon, official_initial)
    local_plan, local_stats = solve_one_start(
        source[None, :, :],
        target[None, :, :],
        args.epsilon,
        local_initial,
        outer_max_iter=1000,
        sinkhorn_max_iter=1000,
        outer_tolerance=1e-9,
        sinkhorn_tolerance=1e-9,
    )
    official_plan = np.asarray(official["ot"], dtype=np.float64)
    local_objective = gw_objective(
        source[None, :, :],
        target[None, :, :],
        local_plan,
        np.full(m, 1.0 / m),
        np.full(n, 1.0 / n),
    )
    result = {
        "scope": "Atanas h=0 single-relation boundary only",
        "source": args.source,
        "target": args.target,
        "source_neurons": m,
        "target_neurons": n,
        "epsilon": args.epsilon,
        "seed": args.seed,
        "pot_version": ot.__version__,
        "initial_plans_elementwise_equal": bool(np.array_equal(official_initial, local_initial)),
        "initial_plan_max_abs_delta": float(np.max(np.abs(official_initial - local_initial))),
        "official_gwtune": {
            "objective": float(official["gw_dist"]),
            "iterations": int(official["cpt"]),
        },
        "paper_reimplementation": {
            "objective": local_objective,
            **local_stats,
        },
        "objective_abs_delta": float(abs(float(official["gw_dist"]) - local_objective)),
        "transport_max_abs_delta": float(np.max(np.abs(official_plan - local_plan))),
        "transport_frobenius_delta": float(np.linalg.norm(official_plan - local_plan)),
        "official_source_scope": "GWTune accepts one source and one target distance matrix; it is not GWOT-MD",
    }
    result["pass"] = bool(
        result["initial_plans_elementwise_equal"]
        and result["objective_abs_delta"] <= 1e-12
        and result["transport_max_abs_delta"] <= 1e-12
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

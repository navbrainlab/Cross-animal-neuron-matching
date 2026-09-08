#!/usr/bin/env python3
"""Strict preflight wrapper for the Appendix-F majority-vote evaluator.

The bundled ``paper_exact_h_selection_majority_vote.py`` performs the actual
1,000 unique 9-teacher experiments, inner leave-one-out h selection, and
v=5/k=5 voting.  This wrapper adds two checks that the generic evaluator cannot
know about:

1. the recording UIDs must be the paper's exact 21 baseline+NeuroPAL cohort;
2. their 21x21 common-label matrix must exactly equal Figure C.1.

These checks catch both the previously used 38-worm cohort and label releases
or uncertain-label rules that change the scoring denominator.  All CLI options
are forwarded to the bundled evaluator.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np


PAPER_COHORT = (
    "2022-06-14-01", "2022-06-14-07", "2022-06-14-13",
    "2022-06-28-01", "2022-06-28-07", "2022-07-15-06",
    "2022-07-15-12", "2022-07-20-01", "2022-07-26-01",
    "2022-08-02-01", "2023-01-09-28", "2023-01-17-01",
    "2023-01-19-01", "2023-01-19-08", "2023-01-19-15",
    "2023-01-19-22", "2023-01-23-01", "2023-01-23-08",
    "2023-01-23-15", "2023-01-23-21", "2023-03-07-01",
)

PAPER_COMMON_LABELS = np.asarray([
    [78,64,64,54,54,50,58,59,52,73,51,59,44,53,32,58,49,57,62,64,56],
    [64,87,71,58,61,54,65,67,53,75,63,67,46,59,33,61,57,59,68,71,60],
    [64,71,95,61,65,56,68,77,54,84,67,72,56,69,43,74,67,66,78,77,66],
    [54,58,61,74,58,48,63,61,52,69,58,64,47,52,36,61,55,59,61,61,55],
    [54,61,65,58,84,49,65,66,51,71,59,62,49,55,38,61,56,58,58,62,56],
    [50,54,56,48,49,69,56,57,46,58,52,55,37,52,33,55,51,49,57,56,54],
    [58,65,68,63,65,56,91,70,54,78,63,68,51,61,37,67,61,59,69,67,62],
    [59,67,77,61,66,57,70,95,55,83,68,72,56,68,43,73,63,60,71,75,63],
    [52,53,54,52,51,46,54,55,67,61,54,57,40,51,28,54,51,51,54,58,50],
    [73,75,84,69,71,58,78,83,61,111,73,78,59,72,47,81,73,73,83,84,70],
    [51,63,67,58,59,52,63,68,54,73,91,72,47,65,43,67,69,63,72,71,60],
    [59,67,72,64,62,55,68,72,57,78,72,90,51,65,37,70,65,67,71,76,62],
    [44,46,56,47,49,37,51,56,40,59,47,51,71,47,40,56,45,49,51,51,44],
    [53,59,69,52,55,52,61,68,51,72,65,65,47,85,39,67,62,57,69,67,59],
    [32,33,43,36,38,33,37,43,28,47,43,37,40,39,60,47,41,38,44,42,35],
    [58,61,74,61,61,55,67,73,54,81,67,70,56,67,47,92,67,65,75,77,64],
    [49,57,67,55,56,51,61,63,51,73,69,65,45,62,41,67,87,63,71,69,55],
    [57,59,66,59,58,49,59,60,51,73,63,67,49,57,38,65,63,87,66,66,55],
    [62,68,78,61,58,57,69,71,54,83,72,71,51,69,44,75,71,66,96,80,67],
    [64,71,77,61,62,56,67,75,58,84,71,76,51,67,42,77,69,66,80,95,65],
    [56,60,66,55,56,54,62,63,50,70,60,62,44,59,35,64,55,55,67,65,82],
], dtype=np.int16)


def find_evaluator() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "paper_exact_h_selection_majority_vote.py",
        here.parent.parent / "reference_files" / "paper_exact_h_selection_majority_vote.py",
        Path.cwd() / "reference_files" / "paper_exact_h_selection_majority_vote.py",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "paper_exact_h_selection_majority_vote.py must be beside this wrapper"
    )


def load_evaluator(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("paper_majority_vote", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_figure_c1(labels_by_worm: dict[str, np.ndarray], worms: list[str]) -> None:
    if tuple(worms) != PAPER_COHORT:
        missing = sorted(set(PAPER_COHORT).difference(worms))
        extra = sorted(set(worms).difference(PAPER_COHORT))
        raise ValueError(
            "Not the paper's exact 21 baseline+NeuroPAL cohort. "
            f"missing={missing}, extra={extra}, observed_order={worms}"
        )
    label_sets = [set(str(x) for x in labels_by_worm[uid] if str(x)) for uid in worms]
    observed = np.asarray(
        [[len(left.intersection(right)) for right in label_sets] for left in label_sets],
        dtype=np.int16,
    )
    if not np.array_equal(observed, PAPER_COMMON_LABELS):
        delta = observed.astype(int) - PAPER_COMMON_LABELS.astype(int)
        mismatches = np.argwhere(delta != 0)
        examples = [
            {
                "row": int(i + 1),
                "column": int(j + 1),
                "observed": int(observed[i, j]),
                "paper": int(PAPER_COMMON_LABELS[i, j]),
                "delta": int(delta[i, j]),
            }
            for i, j in mismatches[:20]
        ]
        raise ValueError(
            "Label denominator is not paper-exact (Figure C.1 mismatch). "
            f"mismatched_cells={len(mismatches)}, max_abs_delta={int(np.max(np.abs(delta)))}, "
            f"observed_diagonal={np.diag(observed).tolist()}, "
            f"paper_diagonal={np.diag(PAPER_COMMON_LABELS).tolist()}, examples={examples}. "
            "Do not report the resulting score as a reproduction of 46%."
        )
    print("  strict preflight: cohort and all 441 Figure-C.1 label counts match", flush=True)


def main() -> None:
    evaluator = load_evaluator(find_evaluator())
    original_load_cache = evaluator.load_cache

    def checked_load_cache(*args: Any, **kwargs: Any) -> Any:
        scores, labels_by_worm, worms = original_load_cache(*args, **kwargs)
        validate_figure_c1(labels_by_worm, worms)
        return scores, labels_by_worm, worms

    evaluator.load_cache = checked_load_cache
    evaluator.main()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
BENCH = ROOT / "benchmark_official"
THIRD = BENCH / "third_party"
OUT = BENCH / "manifests" / "official_repo_audit.json"

SPECS = {
    "fDNC": {
        "path": THIRD / "fdnc_official",
        "url": "https://github.com/XinweiYu/fDNC_Neuron_ID.git",
        "expected": ["src/DNC_predict.py", "requirements.txt", "model"],
        "role": "Position-only cross-animal correspondence; use official inference/model code.",
    },
    "WormID_NWBelegans": {
        "path": THIRD / "wormid_official",
        "url": "https://github.com/focolab/NWBelegans.git",
        "expected": ["NWB_atlas.ipynb", "stat-atlas", ".gitmodules"],
        "role": "Official WormID analysis repository; use its pinned stat-atlas submodule.",
    },
    "Statistical_Atlas": {
        "path": THIRD / "wormid_official" / "stat-atlas",
        "url": "https://github.com/amin-nejat/stat-atlas.git",
        "expected": ["models.py", "utils.py", "demo.py"],
        "role": "Position-only statistical atlas implementation pinned by WormID repository.",
    },
    "NeurPIR": {
        "path": THIRD / "neurpir_official",
        "url": "https://github.com/ww20hust/NeurPIR.git",
        "expected": ["src", "configs", "requirements.txt"],
        "role": "Activity representation; use official encoder/VICReg implementation.",
    },
    "NuCLR": {
        "path": THIRD / "nuclr_official",
        "url": "https://github.com/nerdslab/NuCLR.git",
        "expected": ["src", "configs", "train.py", "eval_scripts"],
        "role": "Activity representation; use official NuCLR model/training/evaluation modules.",
    },
    "MulT": {
        "path": THIRD / "mult_official",
        "url": "https://github.com/yaohungt/Multimodal-Transformer.git",
        "expected": ["modules", "src", "main.py"],
        "role": "Multimodal cross-modal Transformer; adapter may alter I/O/task head only.",
    },
}


def run_ok(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        args,
        cwd=cwd,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def run_maybe(*args: str, cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        args,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode, proc.stdout.strip()


def repo_info(path: Path) -> dict:
    commit = run_ok("git", "rev-parse", "HEAD", cwd=path)

    code, branch = run_maybe(
        "git", "symbolic-ref", "--short", "-q", "HEAD", cwd=path
    )
    if code != 0 or not branch:
        branch = "DETACHED"

    status = run_ok("git", "status", "--porcelain", cwd=path)

    code, superproject = run_maybe(
        "git", "rev-parse", "--show-superproject-working-tree", cwd=path
    )
    is_submodule = bool(code == 0 and superproject)

    return {
        "commit": commit,
        "branch": branch,
        "detached_head": branch == "DETACHED",
        "dirty": bool(status),
        "is_submodule": is_submodule,
        "superproject_worktree": superproject if is_submodule else "",
    }


def main():
    report = {
        "workspace": str(BENCH),
        "policy": {
            "third_party_source_editing_allowed": False,
            "adapter_location": str(BENCH / "adapters"),
            "primary_metric": "direct per-query ranking Top-1 before Hungarian assignment",
            "split_policy": "same locked worm-level outer folds as Ours; validation only for selection",
            "detached_submodule_head_is_valid": True,
        },
        "repositories": {},
    }

    failures = []

    for name, spec in SPECS.items():
        p = spec["path"]
        item = {
            "path": str(p),
            "url": spec["url"],
            "role": spec["role"],
            "exists": p.exists(),
            "expected_files": {},
        }

        if not p.exists():
            failures.append(f"{name}: missing {p}")
            report["repositories"][name] = item
            continue

        try:
            item.update(repo_info(p))
        except Exception as exc:
            failures.append(f"{name}: git audit failed: {exc}")

        for rel in spec["expected"]:
            ok = (p / rel).exists()
            item["expected_files"][rel] = ok
            if not ok:
                failures.append(f"{name}: missing expected {rel}")

        report["repositories"][name] = item

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved: {OUT}")

    if failures:
        print("\nAUDIT FAILURES:")
        for x in failures:
            print(" -", x)
        raise SystemExit(2)

    print("\nAUDIT PASSED: all expected official repositories/files are present.")
    detached = [
        name
        for name, item in report["repositories"].items()
        if item.get("detached_head")
    ]
    if detached:
        print("Detached HEAD repositories/submodules (valid):")
        for name in detached:
            print(" -", name)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Iterable


PRINT_LOCK = threading.Lock()
RUNTIME_OVERRIDES = {
    "dataset_root",
    "output_dir",
    "seed",
    "device",
    "allow_existing_output",
    "variant",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_parser(package_root: Path) -> argparse.ArgumentParser:
    sys.path.insert(0, str(package_root))
    from mprt_net.train import build_parser

    return build_parser()


def source_arguments(run_dir: Path) -> dict[str, Any]:
    args_path = run_dir / "args.json"
    if args_path.is_file():
        return read_json(args_path)
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No args.json or best.pt in {run_dir}")
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    values = checkpoint.get("train_args")
    if not isinstance(values, dict):
        raise ValueError(f"No train_args dictionary in {checkpoint_path}")
    return dict(values)


def complete_arguments(
    parser: argparse.ArgumentParser, source: dict[str, Any]
) -> dict[str, Any]:
    return {
        action.dest: source.get(action.dest, action.default)
        for action in parser._actions
        if action.dest != "help"
    }


def command_from_values(
    parser: argparse.ArgumentParser, values: dict[str, Any]
) -> list[str]:
    command = [sys.executable, "-u", "-m", "mprt_net.train"]
    for action in parser._actions:
        if action.dest == "help":
            continue
        options = [value for value in action.option_strings if value.startswith("--")]
        if not options:
            continue
        option = max(options, key=len)
        value = values[action.dest]
        if action.nargs == 0:
            if bool(value):
                command.append(option)
            continue
        if value is None:
            continue
        command.append(option)
        if isinstance(value, (tuple, list)):
            command.extend(str(item) for item in value)
        else:
            command.append(str(value))
    return command


def run_command(
    command: list[str],
    *,
    cwd: Path,
    gpu: str,
    log_path: Path,
) -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with PRINT_LOCK:
        rendered = (
            shlex.join(command)
            if hasattr(shlex, "join")
            else " ".join(shlex.quote(item) for item in command)
        )
        print(f"[RUN GPU{gpu}] {rendered}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            with PRINT_LOCK:
                print(f"[GPU{gpu}] {line}", end="", flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def parse_csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value")
    return result


def parse_int_csv(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in parse_csv(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def partition(values: list[Any], workers: int) -> list[list[Any]]:
    return [values[index::workers] for index in range(workers)]


def train_spec(
    *,
    source_checkpoint: Path,
    source_values: dict[str, Any],
    dataset_root: Path,
    seed: int,
    variant: str,
) -> dict[str, Any]:
    normalized = {
        key: value for key, value in source_values.items() if key not in RUNTIME_OVERRIDES
    }
    return {
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_sha256": sha256(source_checkpoint),
        "source_train_arguments": normalized,
        "dataset_root": str(dataset_root.resolve()),
        "seed": seed,
        "variant": variant,
    }


def ensure_clean_or_reusable_run(run_dir: Path, spec: dict[str, Any]) -> bool:
    checkpoint = run_dir / "best.pt"
    spec_path = run_dir / "experiment_spec.json"
    if checkpoint.is_file() and spec_path.is_file():
        if read_json(spec_path) != spec:
            raise RuntimeError(f"Existing run has a different specification: {run_dir}")
        return True
    protected = (checkpoint, run_dir / "last.pt", run_dir / "history.jsonl", spec_path)
    if any(path.exists() for path in protected):
        raise FileExistsError(f"Incomplete/stale run exists: {run_dir}")
    return False

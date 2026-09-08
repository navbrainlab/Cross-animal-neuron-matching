#!/usr/bin/env python3
"""Resumably download and verify the Dunn et al. Zenodo processed release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


API_URL = "https://zenodo.org/api/records/17353307"
PRINT_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pickle-root",
        type=Path,
        default=Path("Data/Dunn_001623_intermediate"),
        help="Destination for the 95 individual processed pickle files.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("Data/Zenodo_17353307"),
        help="Destination for metadata and intermediate_datafiles.zip.",
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_metadata(destination: Path) -> dict:
    request = urllib.request.Request(API_URL, headers={"User-Agent": "nuclr-data-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    data = json.loads(payload)
    destination.write_bytes(payload)
    return data


def zenodo_direct_url(key: str) -> str:
    return (
        "https://zenodo.org/records/17353307/files/"
        + urllib.parse.quote(key)
        + "?download=1"
    )


def download_range_chunk(
    url: str, destination: Path, start: int, end: int
) -> None:
    expected_size = end - start + 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 101):
        current_size = destination.stat().st_size if destination.exists() else 0
        if current_size == expected_size:
            return
        if current_size > expected_size:
            destination.unlink()
            current_size = 0
        request_start = start + current_size
        command = [
            shutil.which("curl") or "curl",
            "-fL",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "60",
            "--range",
            f"{request_start}-{end}",
            url,
        ]
        with destination.open("ab") as handle:
            result = subprocess.run(command, stdout=handle, check=False)
        size_after = destination.stat().st_size
        if size_after == expected_size:
            return
        if size_after > expected_size:
            destination.unlink()
        if attempt == 100:
            raise RuntimeError(
                f"range {start}-{end} incomplete after {attempt} attempts; "
                f"last curl exit={result.returncode}, size={size_after}/{expected_size}"
            )
        time.sleep(min(60, 5 * attempt))


def download_archive_ranged(
    entry: dict, destination: Path, range_workers: int = 24, integrity_retries: int = 1
) -> tuple[str, str]:
    expected_size = int(entry["size"])
    expected_md5 = entry["checksum"].removeprefix("md5:")
    if destination.exists():
        if destination.stat().st_size == expected_size and md5sum(destination) == expected_md5:
            return destination.name, "verified-existing"
        destination.unlink()
    assembled = destination.with_name(destination.name + ".part")
    if assembled.exists():
        assembled.unlink()
    parts_root = destination.parent / f".{destination.name}.ranges"
    parts_root.mkdir(parents=True, exist_ok=True)
    chunk_size = (expected_size + range_workers - 1) // range_workers
    chunks: list[tuple[int, int, Path]] = []
    for index in range(range_workers):
        start = index * chunk_size
        if start >= expected_size:
            break
        end = min(expected_size - 1, start + chunk_size - 1)
        chunks.append((start, end, parts_root / f"part_{index:03d}"))

    url = zenodo_direct_url(entry["key"])
    with ThreadPoolExecutor(max_workers=range_workers) as executor:
        futures = {
            executor.submit(download_range_chunk, url, part, start, end): index
            for index, (start, end, part) in enumerate(chunks)
        }
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            with PRINT_LOCK:
                print(
                    f"archive range {completed:02d}/{len(chunks):02d} verified by length",
                    flush=True,
                )

    with assembled.open("wb") as output:
        for start, end, part in chunks:
            if part.stat().st_size != end - start + 1:
                raise RuntimeError(f"range size changed before assembly: {part}")
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    actual_md5 = md5sum(assembled)
    if actual_md5 != expected_md5:
        assembled.unlink()
        shutil.rmtree(parts_root)
        if integrity_retries > 0:
            return download_archive_ranged(
                entry, destination, range_workers, integrity_retries - 1
            )
        raise RuntimeError(
            f"archive MD5 mismatch: expected {expected_md5}, got {actual_md5}"
        )
    os.replace(assembled, destination)
    shutil.rmtree(parts_root)
    return destination.name, "downloaded-ranges-assembled-and-verified"


def download_one(
    entry: dict, destination: Path, checksum_retries: int = 2
) -> tuple[str, str]:
    if entry["key"] == "intermediate_datafiles.zip":
        return download_archive_ranged(entry, destination)
    expected_size = int(entry["size"])
    expected_md5 = entry["checksum"].removeprefix("md5:")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if destination.stat().st_size == expected_size and md5sum(destination) == expected_md5:
            return destination.name, "verified-existing"
        partial = destination.with_name(destination.name + ".part")
        if destination.stat().st_size < expected_size and not partial.exists():
            destination.replace(partial)
        else:
            destination.unlink()

    partial = destination.with_name(destination.name + ".part")
    if partial.exists() and partial.stat().st_size > expected_size:
        partial.unlink()
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for resumable downloads")
    direct_url = zenodo_direct_url(entry["key"])
    command = [
        curl,
        "-fL",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "60",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        direct_url,
    ]
    # Invoke curl again for every retry so --continue-at is recalculated from
    # the latest on-disk length. Curl's internal retry can reopen the output
    # from an earlier offset after a broken response and lose valid progress.
    for attempt in range(1, 101):
        size_before = partial.stat().st_size if partial.exists() else 0
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            break
        size_after = partial.stat().st_size if partial.exists() else 0
        if size_after < size_before:
            raise RuntimeError(
                f"partial file shrank during retry for {entry['key']}: "
                f"{size_before} -> {size_after}"
            )
        if attempt == 100:
            raise RuntimeError(
                f"curl failed after {attempt} resumable attempts "
                f"(last exit {result.returncode}): {entry['key']}"
            )
        time.sleep(min(60, 5 * attempt))
    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        if checksum_retries > 0:
            with PRINT_LOCK:
                print(
                    f"Size mismatch; restarting only {entry['key']} from byte zero "
                    f"({checksum_retries} integrity retries left).",
                    flush=True,
                )
            partial.unlink()
            return download_one(entry, destination, checksum_retries - 1)
        raise RuntimeError(
            f"size mismatch for {entry['key']}: expected {expected_size}, got {actual_size}"
        )
    actual_md5 = md5sum(partial)
    if actual_md5 != expected_md5:
        if checksum_retries > 0:
            with PRINT_LOCK:
                print(
                    f"Checksum mismatch; restarting only {entry['key']} from byte zero "
                    f"({checksum_retries} checksum retries left).",
                    flush=True,
                )
            partial.unlink()
            return download_one(entry, destination, checksum_retries - 1)
        raise RuntimeError(
            f"MD5 mismatch for {entry['key']}: expected {expected_md5}, got {actual_md5}"
        )
    os.replace(partial, destination)
    return destination.name, "downloaded-and-verified"


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    pickle_root = args.pickle_root.expanduser().resolve()
    archive_root = args.archive_root.expanduser().resolve()
    pickle_root.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)

    metadata_path = archive_root / "record_17353307.json"
    metadata = fetch_metadata(metadata_path)
    entries = metadata["files"]
    pickle_entries = sorted(
        (entry for entry in entries if entry["key"].endswith(".pkl")),
        key=lambda entry: entry["key"],
    )
    zip_entries = [entry for entry in entries if entry["key"] == "intermediate_datafiles.zip"]
    if len(pickle_entries) != 95 or len(zip_entries) != 1:
        raise RuntimeError(
            f"Unexpected Zenodo contents: {len(pickle_entries)} pickle(s), {len(zip_entries)} zip(s)"
        )

    jobs = [(entry, pickle_root / entry["key"]) for entry in pickle_entries]
    jobs.append((zip_entries[0], archive_root / "intermediate_datafiles.zip"))
    print(
        f"Zenodo record {metadata['id']}: {len(pickle_entries)} pickles + 1 archive, "
        f"{sum(entry['size'] for entry, _ in jobs) / 2**30:.3f} GiB",
        flush=True,
    )

    failures: list[str] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, entry, destination): entry["key"]
            for entry, destination in jobs
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                name, status = future.result()
                completed += 1
                with PRINT_LOCK:
                    print(f"[{completed:02d}/{len(jobs)}] {status}: {name}", flush=True)
            except Exception as exc:
                failures.append(f"{key}: {exc}")
                with PRINT_LOCK:
                    print(f"FAILED: {key}: {exc}", flush=True)

    if failures:
        raise RuntimeError("Download incomplete:\n" + "\n".join(failures))
    print(f"All {len(pickle_entries)} pickles and the archive passed MD5 verification.")


if __name__ == "__main__":
    main()

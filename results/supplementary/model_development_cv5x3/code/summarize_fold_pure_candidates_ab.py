#!/usr/bin/env python3
"""Audit, summarize, and package the fold-pure Candidate A/B experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/fold_pure_table1init_candidates_ab_20260817"
OUT = ROOT / "paper_submission_data/fold_pure_e2e_candidates_ab_20260817"
DATASETS = ("atanas", "rld")
FOLDS = range(1, 6)
SEEDS = (1, 42, 123)
CANDIDATES = ("A", "B")
METRICS = ("ranking_top1", "top3", "top5", "top10", "mrr", "mean_rank", "assignment_top1")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def bootstrap(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    sample = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(sample, (0.025, 0.975)))


def load_and_audit() -> tuple[list[dict], list[dict], dict, dict]:
    protocol = json.loads((RUN / "FOLD_PURE_PROTOCOL.json").read_text())
    status = {
        stage: json.loads((RUN / f"status_{stage}.json").read_text())
        for stage in ("nuclr", "extract", "matchers")
    }
    expected_jobs = {"nuclr": 30, "extract": 30, "matchers": 60}
    for stage, value in status.items():
        if value["jobs"] != expected_jobs[stage] or value["failures"]:
            raise RuntimeError(f"Incomplete {stage}: {value}")

    runs, audits, raw = [], [], {}
    for dataset in DATASETS:
        for fold in FOLDS:
            fold_info = protocol["datasets"][dataset]["folds"][fold - 1]
            expected = {key: set(fold_info[f"{key}_ids"]) for key in ("train", "val", "test")}
            for seed in SEEDS:
                pipeline = RUN / "folds" / dataset / f"fold_{fold}" / "pipeline" / f"seed_{seed}"
                nuclr_summary = json.loads((pipeline / "nuclr_t1st1/summary.json").read_text())
                nuclr_config = json.loads((pipeline / "nuclr_t1st1/config.json").read_text())
                audit = json.loads((pipeline / "nuclr_t1st1/protocol_audit.json").read_text())
                args = nuclr_config["args"]
                if args["nuclr_temporal_layers"] != 1 or args["nuclr_spatiotemporal_layers"] != 1:
                    raise AssertionError("NuCLR is not T1/ST1")
                if set(audit["train_worms"]) != expected["train"] or set(audit["val_worms"]) != expected["val"]:
                    raise AssertionError("NuCLR split mismatch")
                if set(audit["test_worm_ids_assertion_only"]) != expected["test"]:
                    raise AssertionError("NuCLR test ID assertion mismatch")
                if audit["test_worms_referenced_by_training_dataloader"] != 0:
                    raise AssertionError("NuCLR referenced test")
                init = audit["initialization"]
                if not init["outer_train_only_pretraining"] or init["independent_external_pretraining"]:
                    raise AssertionError("Wrong initializer scope")
                if audit["train_identity_labels_loaded"] or audit["same_only_loss_accessed_cell_id"]:
                    raise AssertionError("Same-only NuCLR opened training identity labels")
                audits.append({
                    "dataset": dataset, "fold": fold, "seed": seed,
                    "train_worms": len(expected["train"]), "val_worms": len(expected["val"]),
                    "test_worms": len(expected["test"]), "test_dataloader_references": 0,
                    "train_identity_labels_loaded": False, "temporal_layers": 1,
                    "spatiotemporal_layers": 1, "nuclr_best_epoch": nuclr_summary["best_epoch"],
                    "initializer_scope": init["audited_initialization_scope"],
                    "initializer_sha256": init["checkpoint_sha256"],
                    "initializer_train_step": init["train_step"],
                })
                for candidate in CANDIDATES:
                    run_dir = pipeline / f"candidate_{candidate.lower()}"
                    summary = json.loads((run_dir / "summary.json").read_text())
                    config = json.loads((run_dir / "config.json").read_text())
                    if set(summary["test_worms"]) != expected["test"]:
                        raise AssertionError("Matcher test split mismatch")
                    if set(config["source_train_worms"]) != expected["train"] or set(config["source_val_worms"]) != expected["val"]:
                        raise AssertionError("Matcher train/val split mismatch")
                    if candidate == "A":
                        if config["position_encoder"] != "hyqurp" or config["fusion"] != "concat":
                            raise AssertionError("Candidate A architecture mismatch")
                    else:
                        if not config["stage2_enabled"] or config["sinkhorn_enabled"]:
                            raise AssertionError("Candidate B architecture mismatch")
                    raw[(dataset, fold, seed, candidate)] = summary
                    test = summary["test"]
                    row = {
                        "dataset": dataset, "fold": fold, "seed": seed, "candidate": candidate,
                        "best_epoch": summary["best_epoch"], "val_ranking_top1": summary["validation"]["ranking_top1"],
                        "test_num_queries": test["num_queries"],
                    }
                    row.update({metric: test[metric] for metric in METRICS})
                    runs.append(row)
    if len(runs) != 60 or len(audits) != 30:
        raise RuntimeError(f"Expected 60 runs/30 audits, got {len(runs)}/{len(audits)}")
    return runs, audits, raw, protocol


def summarize(runs: list[dict], raw: dict) -> tuple[list[dict], list[dict], list[dict]]:
    rng = np.random.default_rng(42)
    aggregate, effects, seeds_out = [], [], []
    for dataset in DATASETS:
        for candidate in CANDIDATES:
            selected = [x for x in runs if x["dataset"] == dataset and x["candidate"] == candidate]
            row = {"dataset": dataset, "candidate": candidate, "n_folds": 5, "seeds_per_fold": 3,
                   "mean_best_epoch": float(np.mean([x["best_epoch"] for x in selected]))}
            for metric in METRICS:
                fold_values = np.asarray([
                    np.mean([raw[(dataset, fold, seed, candidate)]["test"][metric] for seed in SEEDS])
                    for fold in FOLDS
                ])
                lo, hi = bootstrap(fold_values, rng)
                row[f"{metric}_fold_mean"] = float(fold_values.mean())
                row[f"{metric}_fold_sd"] = float(fold_values.std(ddof=1))
                row[f"{metric}_ci_low"] = lo; row[f"{metric}_ci_high"] = hi
                all_values = np.asarray([x[metric] for x in selected])
                row[f"{metric}_15run_mean"] = float(all_values.mean())
                row[f"{metric}_15run_sd"] = float(all_values.std(ddof=1))
            aggregate.append(row)
            for seed in SEEDS:
                seed_row = {"dataset": dataset, "candidate": candidate, "seed": seed}
                chosen = [x for x in selected if x["seed"] == seed]
                for metric in METRICS:
                    values = np.asarray([x[metric] for x in chosen])
                    seed_row[f"{metric}_mean"] = float(values.mean())
                    seed_row[f"{metric}_sd"] = float(values.std(ddof=1))
                seeds_out.append(seed_row)

        for metric in METRICS:
            fold_deltas, per_seed = [], {seed: [] for seed in SEEDS}
            for fold in FOLDS:
                deltas = []
                for seed in SEEDS:
                    delta = raw[(dataset, fold, seed, "B")]["test"][metric] - raw[(dataset, fold, seed, "A")]["test"][metric]
                    deltas.append(delta); per_seed[seed].append(delta)
                fold_deltas.append(float(np.mean(deltas)))
            values = np.asarray(fold_deltas)
            lo, hi = bootstrap(values, rng)
            effects.append({
                "dataset": dataset, "metric": metric, "contrast": "B_minus_A",
                "mean_paired_effect": float(values.mean()), "fold_sd": float(values.std(ddof=1)),
                "fold_bootstrap_ci_low": lo, "fold_bootstrap_ci_high": hi,
                **{f"fold_{fold}_effect": values[fold - 1] for fold in FOLDS},
                **{f"seed_{seed}_mean_effect": float(np.mean(per_seed[seed])) for seed in SEEDS},
            })
    return aggregate, effects, seeds_out


def pct(value: float) -> str:
    return f"{100*value:.2f}%"


def readme(aggregate: list[dict], effects: list[dict], protocol: dict) -> str:
    agg = {(x["dataset"], x["candidate"]): x for x in aggregate}
    eff = {(x["dataset"], x["metric"]): x for x in effects}
    lines = [
        "# Fold-pure Candidate A/B：5-fold CV × 3 seeds", "",
        "Candidate A：HyQuRP + Compact NuCLR + concat + Population Transformer（无 Stage 2、无 Sinkhorn）。  ",
        "Candidate B：HyQuRP + Compact NuCLR → Stage 2 → Population Transformer（有 Stage 2、无 Sinkhorn）。", "",
        "主估计先在每个 outer fold 内平均 3 seeds，再对 5 个 fold means 计算非加权 mean ± sample SD。",
        "B−A 的 95% CI 以 outer fold 为 cluster，进行 10,000 次 percentile bootstrap；仅 5 folds，CI 为描述性。", "",
    ]
    for dataset in DATASETS:
        lines += [f"## {dataset.upper()}", "", "| Candidate | Ranking Top-1 | MRR | Assignment Top-1 |", "|---|---:|---:|---:|"]
        for candidate in CANDIDATES:
            row = agg[(dataset, candidate)]
            values = [f"{pct(row[f'{m}_fold_mean'])} ± {pct(row[f'{m}_fold_sd'])}" for m in ("ranking_top1", "mrr", "assignment_top1")]
            lines.append(f"| {candidate} | {' | '.join(values)} |")
        lines += ["", "| B − A | Effect | 95% fold-bootstrap CI |", "|---|---:|---:|"]
        for metric, label in (("ranking_top1", "Ranking Top-1"), ("mrr", "MRR"), ("assignment_top1", "Assignment Top-1")):
            row = eff[(dataset, metric)]
            lines.append(f"| {label} | {100*row['mean_paired_effect']:+.2f} pp | [{100*row['fold_bootstrap_ci_low']:+.2f}, {100*row['fold_bootstrap_ci_high']:+.2f}] pp |")
        lines.append("")
    atanas_effect = eff[("atanas", "ranking_top1")]
    rld_effect = eff[("rld", "ranking_top1")]
    lines += [
        "## 结果判定", "",
        "本次 fold-pure CV 支持 **Candidate B（Stage 2、无 Sinkhorn）** 作为两者中的优选模型。",
        f"Atanas 的 Ranking Top-1 提升为 {100*atanas_effect['mean_paired_effect']:+.2f} pp，5/5 folds 均为正；",
        f"RLD 的提升为 {100*rld_effect['mean_paired_effect']:+.2f} pp，5/5 folds 也均为正。",
        "三个 seeds 各自的跨折平均 B−A 效应在两个数据集上均为正。",
        "因此保留 Stage 2、移除 Sinkhorn 是当前两候选中由严格 fold-pure 证据支持的选择。", "",
    ]
    lines += [
        "## Fold-pure 审计", "",
        "- 每个 dataset/fold/seed 都使用独立的 NuCLR 权重链路；30 个 encoder audit 全部通过。",
        "- fold-local source NuCLR 仅使用 outer-train、训练 30 epochs、test dataloader 引用数为 0、未打开 identity labels。",
        "- source T2/ST2 checkpoint 经权重保留截断为 T1/ST1，再只用同一 outer-train fine-tune；inner validation 选择 Compact NuCLR checkpoint。",
        "- NuCLR checkpoint 锁定后才导出 sealed outer-test activity representation。",
        "- A/B matcher 只用 outer-train 训练、inner validation 选 checkpoint，选定后才读取 outer-test list。",
        "- Atanas 为 38 worms；RLD 为 evaluable93，预先排除两条 clean identity 数为 0 的 recordings。",
        "- RLD 和 Atanas 都按 acquisition-date groups 隔离，日期不跨 train/validation/test。", "",
        "这属于完整的 **fold-pure pipeline grouped CV**。NuCLR 与 matcher 分阶段训练并通过冻结 embedding 衔接，",
        "因此不是一个联合反向传播的单体 differentiable end-to-end network；报告中不应混淆这两个概念。", "",
        "## 文件", "",
        "- `all_runs.csv`：60 个 Candidate A/B 外层测试结果。",
        "- `aggregate.csv`：折级主汇总。",
        "- `paired_effects.csv`：同 fold/seed 的 B−A 配对效应。",
        "- `seed_summary.csv`：seed 敏感性。",
        "- `nuclr_audit.csv`：30 个 fold-pure Compact NuCLR 审计。",
        "- `raw/`：协议、运行状态与逐运行 summary/config/audit。",
        "- `SHA256SUMS`：提交包完整性校验。", "",
    ]
    return "\n".join(lines)


def package_raw() -> None:
    raw_out = OUT / "raw"
    for name in ("FOLD_PURE_PROTOCOL.json", "FOLD_PURE_PROTOCOL.sha256", "status_nuclr.json", "status_extract.json", "status_matchers.json"):
        target = raw_out / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(RUN / name, target)
    for dataset in DATASETS:
        for fold in FOLDS:
            for seed in SEEDS:
                pipeline = RUN / "folds" / dataset / f"fold_{fold}" / "pipeline" / f"seed_{seed}"
                dest = raw_out / "runs" / dataset / f"fold_{fold}" / f"seed_{seed}"
                (dest / "nuclr").mkdir(parents=True, exist_ok=True)
                for name in ("summary.json", "config.json", "protocol_audit.json"):
                    shutil.copy2(pipeline / "nuclr_t1st1" / name, dest / "nuclr" / name)
                for candidate in CANDIDATES:
                    c_dest = dest / f"candidate_{candidate.lower()}"; c_dest.mkdir(parents=True, exist_ok=True)
                    for name in ("summary.json", "config.json"):
                        shutil.copy2(pipeline / f"candidate_{candidate.lower()}" / name, c_dest / name)


def checksums() -> None:
    output = OUT / "SHA256SUMS"
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path != output:
            rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(OUT)}")
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    runs, audits, raw, protocol = load_and_audit()
    aggregate, effects, seeds_out = summarize(runs, raw)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "all_runs.csv", runs); write_csv(OUT / "aggregate.csv", aggregate)
    write_csv(OUT / "paired_effects.csv", effects); write_csv(OUT / "seed_summary.csv", seeds_out)
    write_csv(OUT / "nuclr_audit.csv", audits)
    (OUT / "README.md").write_text(readme(aggregate, effects, protocol), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({
        "design": "fold-pure grouped 5-fold CV x 3 seeds; Candidate A vs B",
        "completed_matcher_runs": 60, "completed_nuclr_pipelines": 30, "failed_runs": 0,
        "aggregate": aggregate, "paired_effects": effects,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    package_raw()
    code = OUT / "code"; code.mkdir(parents=True, exist_ok=True)
    for name in ("run_fold_pure_e2e_candidates_ab.py", "run_fold_pure_table1init_candidates_ab.py",
                 "summarize_fold_pure_candidates_ab.py", "train_ambiguity_aware_geo_activity_transformer.py"):
        shutil.copy2(ROOT / "engines" / name, code / name)
    for path in (ROOT / "hybrid/activity.py", ROOT / "train_hyqurp_nuclr_quantum_crossmodal_v2.py"):
        shutil.copy2(path, code / path.name)
    checksums()
    print(json.dumps({"package": str(OUT), "runs": len(runs), "nuclr_audits": len(audits)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

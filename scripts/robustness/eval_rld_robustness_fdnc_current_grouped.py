#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, re, subprocess, sys
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np

ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
CORR_ROOT = ROOT / "runs/rld_robustness_cv5_seed42_v2/corruptions"
CLEAN_ROOT = ROOT / "runs/fdnc_current_grouped_cv_v2/rld"
OUT_ROOT = ROOT / "runs/rld_robustness_cv5_seed42_v2/results/fdnc"
EVAL = ROOT / "scripts/fair_identity/evaluate_train_reference_ensemble.py"
CANONICAL_QUERY_MANIFEST = (
    ROOT / "runs/unified_main_benchmark_cv5_seed42_v2/canonical_query_manifest.csv"
)

COND_RE = re.compile(r"^(coord_noise|missing|outlier)_l([0-9]+(?:\.[0-9]+)?)_p([0-9]+)$")

def loadj(p: Path) -> dict:
    if not p.is_file(): raise FileNotFoundError(p)
    return json.loads(p.read_text())

def metric_block(report: dict) -> dict:
    return report["metrics"]["template_score"]

def clean_paths(fold: int):
    base = CLEAN_ROOT / f"fold{fold}/seed42"
    ckpt = base / "selected/best.pt"
    metrics = base / "outer_test_medoid_template_v1/metrics.json"
    if not ckpt.is_file(): raise FileNotFoundError(ckpt)
    if not metrics.is_file(): raise FileNotFoundError(metrics)
    return base, ckpt, metrics

def discover(fold: int, kinds: set[str]):
    root = CORR_ROOT / f"fold{fold}"
    rows=[]
    for p in sorted(root.iterdir()):
        if not p.is_dir(): continue
        m=COND_RE.match(p.name)
        if not m: continue
        kind, sev, ps = m.groups()
        if kind not in kinds: continue
        rows.append(dict(fold=fold, kind=kind, severity=float(sev), perturbation_seed=int(ps), name=p.name, root=p))
    order={"coord_noise":0,"missing":1,"outlier":2}
    rows.sort(key=lambda r:(order[r["kind"]],r["severity"],r["perturbation_seed"]))
    return rows

def run_condition(fold:int, row:dict, ckpt:Path, force:bool, device:str):
    dst=OUT_ROOT/f"fold{fold}"/row["name"]
    report=dst/"metrics.json"
    if report.is_file() and not force:
        return loadj(report)
    dst.mkdir(parents=True,exist_ok=True)
    cache=dst/"score_cache"
    cmd=[
        sys.executable,"-u",str(EVAL),
        "--method","fdnc",
        "--method-label","fDNC official fine-tuned (current grouped CV)",
        "--fold-root",str(row["root"]),
        "--checkpoint",str(ckpt),
        "--normalization","zscore",
        "--device",device,
        "--workers","1",
        "--cache-dir",str(cache),
        "--output-dir",str(dst),
    ]
    print("[RUN]"," ".join(cmd),flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)
    return loadj(report)

def exact_clean_guard(saved:dict,replay:dict,fold:int):
    a=metric_block(saved); b=metric_block(replay)
    keys=("queries","top1","top5","mrr","hungarian_accuracy")
    bad=[]
    for k in keys:
        av=a[k]; bv=b[k]
        d=abs(float(av)-float(bv))
        print(f"[CLEAN GUARD] fold{fold} {k}: saved={av} replay={bv} diff={d:.3e}")
        if (k=="queries" and int(av)!=int(bv)) or (k!="queries" and d>1e-12):
            bad.append((k,av,bv,d))
    ta=saved["template_selection"]["template_uid"]
    tb=replay["template_selection"]["template_uid"]
    print(f"[CLEAN GUARD] fold{fold} template: saved={ta} replay={tb}")
    if ta!=tb: bad.append(("template_uid",ta,tb,None))
    ca=str(Path(saved["checkpoint"]).resolve())
    cb=str(Path(replay["checkpoint"]).resolve())
    if ca!=cb: bad.append(("checkpoint",ca,cb,None))
    if bad:
        raise RuntimeError(f"fold{fold} clean replay mismatch: {bad}")
    print(f"[CLEAN GUARD EXACT] fold{fold}",flush=True)

def canonical_queries(path: Path):
    frame = pd.read_csv(path)
    frame = frame[frame["dataset"] == "rld"]
    return {
        (int(fold), str(uid)): set(zip(group["node_index"].astype(int), group["identity"].astype(str)))
        for (fold, uid), group in frame.groupby(["fold", "uid"])
    }


def canonical_query_count(fold: int, root: Path, canonical: dict) -> int:
    condition = loadj(root / "CORRUPTION.json")
    total = 0
    for record in condition["files"]:
        uid = str(record["recording_uid"])
        kept = set(int(value) for value in record["kept_source_indices"])
        total += sum(
            int(source_row in kept)
            for source_row, _ in canonical.get((fold, uid), set())
        )
    return total


def result_row(fold,row,report,clean,canonical,clean_canonical_q):
    m=metric_block(report); cm=metric_block(clean)
    native_q=int(m["queries"]); clean_native_q=int(cm["queries"])
    q=canonical_query_count(fold,row["root"],canonical)
    correct=float(m["top1"])*native_q
    top5_correct=float(m["top5"])*native_q
    rr_sum=float(m["mrr"])*native_q
    hungarian_correct=float(m["hungarian_accuracy"])*native_q
    cov=q/clean_canonical_q if clean_canonical_q else float("nan")
    return {
        "fold":fold,"kind":row["kind"],"severity":row["severity"],
        "perturbation_seed":row["perturbation_seed"],
        "queries":q,"clean_queries":clean_canonical_q,
        "reference_covered_queries":native_q,
        "clean_reference_covered_queries":clean_native_q,
        "reference_coverage":native_q/q if q else float("nan"),
        "top1":correct/q if q else float("nan"),
        "top5":top5_correct/q if q else float("nan"),
        "mrr":rr_sum/q if q else float("nan"),
        "hungarian":hungarian_correct/q if q else float("nan"),
        "conditional_top1_reference_covered":float(m["top1"]),
        "coverage_vs_clean":cov,
        "effective_top1":correct/clean_canonical_q if clean_canonical_q else float("nan"),
        "effective_top5":top5_correct/clean_canonical_q if clean_canonical_q else float("nan"),
        "effective_hungarian":hungarian_correct/clean_canonical_q if clean_canonical_q else float("nan"),
        "template_uid":report["template_selection"]["template_uid"],
        "checkpoint":report["checkpoint"],
    }

def summarize(rows):
    OUT_ROOT.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows)
    df.to_csv(OUT_ROOT/"fdnc_all_cells.csv",index=False)
    metrics=["top1","top5","mrr","hungarian","coverage_vs_clean","effective_top1","effective_top5","effective_hungarian"]
    fold_df=df.groupby(["kind","severity","fold"],as_index=False)[metrics].mean()
    fold_df.to_csv(OUT_ROOT/"fdnc_fold_level.csv",index=False)
    out=[]
    for (kind,sev),g in fold_df.groupby(["kind","severity"]):
        r={"kind":kind,"severity":float(sev),"folds":int(len(g))}
        for k in metrics:
            v=g[k].values.astype(float)
            r[k+"_mean"]=float(v.mean())
            r[k+"_sd"]=float(v.std(ddof=1)) if len(v)>1 else 0.0
        out.append(r)
    sm=pd.DataFrame(out).sort_values(["kind","severity"])
    sm.to_csv(OUT_ROOT/"fdnc_macro_summary.csv",index=False)
    print("\n"+"="*100)
    print("fDNC CURRENT-GROUPED ROBUSTNESS MACRO SUMMARY")
    print("="*100)
    for _,r in sm.iterrows():
        print(f"{r['kind']:12s} level={r['severity']:.2f} "
              f"Top1={100*r['top1_mean']:.2f}±{100*r['top1_sd']:.2f}% "
              f"Hung={100*r['hungarian_mean']:.2f}±{100*r['hungarian_sd']:.2f}% "
              f"Cov={100*r['coverage_vs_clean_mean']:.2f}±{100*r['coverage_vs_clean_sd']:.2f}% "
              f"EffTop1={100*r['effective_top1_mean']:.2f}±{100*r['effective_top1_sd']:.2f}%")

def main():
    global CORR_ROOT, OUT_ROOT, CANONICAL_QUERY_MANIFEST
    ap=argparse.ArgumentParser()
    ap.add_argument("--folds",default="0")
    ap.add_argument("--kinds",default="coord_noise")
    ap.add_argument("--force",action="store_true")
    ap.add_argument("--corruption-root",type=Path,default=CORR_ROOT)
    ap.add_argument("--out-root",type=Path,default=OUT_ROOT)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--canonical-query-manifest",type=Path,default=CANONICAL_QUERY_MANIFEST)
    args=ap.parse_args()
    CORR_ROOT=args.corruption_root.resolve()
    OUT_ROOT=args.out_root.resolve()
    CANONICAL_QUERY_MANIFEST=args.canonical_query_manifest.resolve()
    canonical=canonical_queries(CANONICAL_QUERY_MANIFEST)
    folds=[int(x) for x in args.folds.split(",") if x.strip()]
    kinds={x.strip() for x in args.kinds.split(",") if x.strip()}
    allrows=[]
    for fold in folds:
        base,ckpt,clean_metrics_path=clean_paths(fold)
        clean=loadj(clean_metrics_path)
        clean_canonical_q=sum(
            len(query_rows)
            for (candidate_fold, _uid), query_rows in canonical.items()
            if candidate_fold == fold
        )

        # Always replay severity0 through corruption materialization first.
        zero=CORR_ROOT/f"fold{fold}/coord_noise_l0.00_p0"
        if not zero.is_dir(): raise FileNotFoundError(zero)
        zr=dict(fold=fold,kind="coord_noise",severity=0.0,perturbation_seed=0,name=zero.name,root=zero)
        replay=run_condition(fold,zr,ckpt,args.force,args.device)
        exact_clean_guard(clean,replay,fold)
        allrows.append(result_row(fold,zr,replay,clean,canonical,clean_canonical_q))

        for row in discover(fold,kinds):
            if row["kind"]=="coord_noise" and abs(row["severity"])<1e-15 and row["perturbation_seed"]==0:
                continue
            rep=run_condition(fold,row,ckpt,args.force,args.device)
            if rep["template_selection"]["template_uid"] != clean["template_selection"]["template_uid"]:
                raise RuntimeError(
                    f"fold{fold} {row['name']}: reference changed from "
                    f"{clean['template_selection']['template_uid']} to "
                    f"{rep['template_selection']['template_uid']}"
                )
            allrows.append(result_row(
                fold,row,rep,clean,canonical,clean_canonical_q
            ))
    summarize(allrows)
    print("\nCOMPLETE")
    print("results:",OUT_ROOT)
    print("macro:",OUT_ROOT/"fdnc_macro_summary.csv")

if __name__=="__main__":
    main()

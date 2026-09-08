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

def result_row(fold,row,report,clean):
    m=metric_block(report); cm=metric_block(clean)
    q=int(m["queries"]); cq=int(cm["queries"])
    cov=q/cq if cq else float("nan")
    return {
        "fold":fold,"kind":row["kind"],"severity":row["severity"],
        "perturbation_seed":row["perturbation_seed"],
        "queries":q,"clean_queries":cq,
        "top1":float(m["top1"]),"top5":float(m["top5"]),
        "mrr":float(m["mrr"]),"hungarian":float(m["hungarian_accuracy"]),
        "coverage_vs_clean":cov,
        "effective_top1":float(m["top1"])*cov,
        "effective_top5":float(m["top5"])*cov,
        "effective_hungarian":float(m["hungarian_accuracy"])*cov,
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
            v=g[k].to_numpy(float)
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
    global CORR_ROOT, OUT_ROOT
    ap=argparse.ArgumentParser()
    ap.add_argument("--folds",default="0")
    ap.add_argument("--kinds",default="coord_noise")
    ap.add_argument("--force",action="store_true")
    ap.add_argument("--corruption-root",type=Path,default=CORR_ROOT)
    ap.add_argument("--out-root",type=Path,default=OUT_ROOT)
    ap.add_argument("--device",default="cuda")
    args=ap.parse_args()
    CORR_ROOT=args.corruption_root.resolve()
    OUT_ROOT=args.out_root.resolve()
    folds=[int(x) for x in args.folds.split(",") if x.strip()]
    kinds={x.strip() for x in args.kinds.split(",") if x.strip()}
    allrows=[]
    for fold in folds:
        base,ckpt,clean_metrics_path=clean_paths(fold)
        clean=loadj(clean_metrics_path)

        # Always replay severity0 through corruption materialization first.
        zero=CORR_ROOT/f"fold{fold}/coord_noise_l0.00_p0"
        if not zero.is_dir(): raise FileNotFoundError(zero)
        zr=dict(fold=fold,kind="coord_noise",severity=0.0,perturbation_seed=0,name=zero.name,root=zero)
        replay=run_condition(fold,zr,ckpt,args.force,args.device)
        exact_clean_guard(clean,replay,fold)
        allrows.append(result_row(fold,zr,replay,clean))

        for row in discover(fold,kinds):
            if row["kind"]=="coord_noise" and abs(row["severity"])<1e-15 and row["perturbation_seed"]==0:
                continue
            rep=run_condition(fold,row,ckpt,args.force,args.device)
            allrows.append(result_row(fold,row,rep,clean))
    summarize(allrows)
    print("\nCOMPLETE")
    print("results:",OUT_ROOT)
    print("macro:",OUT_ROOT/"fdnc_macro_summary.csv")

if __name__=="__main__":
    main()

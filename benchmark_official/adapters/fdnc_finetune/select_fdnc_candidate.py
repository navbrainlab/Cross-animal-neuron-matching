#!/usr/bin/env python3
"""Select one fDNC fine-tuning candidate using validation only."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--candidate-root",type=Path,required=True)
    p.add_argument("--selected-dir",type=Path,required=True)
    args=p.parse_args()

    rows=[]
    for d in sorted(args.candidate_root.glob("*")):
        f=d/"final_results.json"
        b=d/"best.pt"
        if not f.is_file() or not b.is_file():
            continue
        x=json.loads(f.read_text())
        v=x["best_validation"]
        rows.append({
            "candidate":d.name,
            "path":str(d.resolve()),
            "top1":float(v["ranking_top1"]),
            "mrr":float(v["mrr"]),
            "best_epoch":int(x["best_epoch"]),
        })
    if not rows:
        raise RuntimeError(f"No completed candidates under {args.candidate_root}")

    rows.sort(key=lambda r:(-r["top1"],-r["mrr"],r["candidate"]))
    best=rows[0]
    args.selected_dir.mkdir(parents=True,exist_ok=True)
    shutil.copy2(Path(best["path"])/"best.pt",args.selected_dir/"best.pt")
    payload={
        "selection_split":"validation only",
        "primary_selection_metric":"ranking_top1",
        "tie_breaker":"mrr",
        "selected":best,
        "candidates":rows,
        "test_data_used":False,
    }
    (args.selected_dir/"selection.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2))

if __name__=="__main__":
    main()

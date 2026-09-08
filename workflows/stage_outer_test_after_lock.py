#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path

def read_ids(path: Path) -> list[str]:
    return [Path(x.strip()).stem for x in path.read_text().splitlines()
            if x.strip() and not x.lstrip().startswith("#")]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--test-ids",type=Path,required=True)
    p.add_argument("--raw-root",type=Path,required=True)
    p.add_argument("--train-root",type=Path,required=True)
    p.add_argument("--val-root",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--expected-worms",type=int,required=True)
    a=p.parse_args()
    if not a.checkpoint.is_file():
        raise FileNotFoundError(f"Final fold best.pt must exist before test is opened: {a.checkpoint}")
    wanted=read_ids(a.test_ids)
    if len(wanted)!=a.expected_worms or len(set(wanted))!=len(wanted):
        raise RuntimeError(f"expected {a.expected_worms} unique test IDs, got {len(wanted)}/{len(set(wanted))}")
    train={x.stem for x in a.train_root.rglob("*.npz")}
    val={x.stem for x in a.val_root.rglob("*.npz")}
    if set(wanted)&train or set(wanted)&val:
        raise RuntimeError("outer-test overlap with fold train/val")
    by_stem={}
    for x in a.raw_root.rglob("*.npz"):
        by_stem.setdefault(x.stem,[]).append(x.resolve())
    bad={w:[str(x) for x in by_stem.get(w,[])] for w in wanted if len(by_stem.get(w,[]))!=1}
    if bad:
        raise RuntimeError(f"outer-test ID resolution failed under {a.raw_root}: {bad}")
    a.output_root.mkdir(parents=True,exist_ok=True)
    for x in a.output_root.glob("*.npz"):
        x.unlink()
    rows=[]
    for w in wanted:
        src=by_stem[w][0]
        dst=a.output_root/f"{w}.npz"
        os.symlink(src,dst)
        rows.append({"worm_id":w,"source":str(src),"staged":str(dst)})
    payload={"outer_test_opened_after_final_checkpoint":True,
             "checkpoint":str(a.checkpoint.resolve()),"worms":rows}
    (a.output_root/"manifest.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2))
if __name__=="__main__":
    main()

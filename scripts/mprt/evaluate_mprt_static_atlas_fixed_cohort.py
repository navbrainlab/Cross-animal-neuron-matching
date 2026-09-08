#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch

@dataclass
class Totals:
    queries:int=0; covered_queries:int=0; top1:int=0; top5:int=0
    reciprocal_rank_sum:float=0.0; top1_with_dustbin:int=0; dustbin_top1:int=0
    hungarian_queries:int=0; hungarian_correct:int=0
    def update(self,o):
        for k in self.__dataclass_fields__: setattr(self,k,getattr(self,k)+getattr(o,k))
    def metrics(self):
        q=max(self.queries,1); hq=max(self.hungarian_queries,1); cq=max(self.covered_queries,1)
        return {
            "queries":self.queries,"covered_queries":self.covered_queries,
            "candidate_coverage":self.covered_queries/q,
            "top1_real":self.top1/q,"top5_real":self.top5/q,
            "mrr_real":self.reciprocal_rank_sum/q,
            "covered_only_top1":self.top1/cq if self.covered_queries else 0.0,
            "covered_only_top5":self.top5/cq if self.covered_queries else 0.0,
            "covered_only_mrr":self.reciprocal_rank_sum/cq if self.covered_queries else 0.0,
            "top1_with_dustbin":self.top1_with_dustbin/q,
            "dustbin_top1_rate":self.dustbin_top1/q,
            "hungarian_queries":self.hungarian_queries,
            "hungarian_accuracy":self.hungarian_correct/hq,
        }

def write_json(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    tmp.replace(path)

def canonical_queries(sample):
    from mprt_net.data import unique_identity_map
    pairs=sorted(((str(i),int(n)) for i,n in unique_identity_map(sample).items()),key=lambda x:x[1])
    return pairs

def eval_sample(output,sample,identity_to_slot,slot_to_identity,uid,source_path):
    probs=output.row_conditional.detach(); real=probs[:,:-1]
    pairs=canonical_queries(sample); t=Totals(); rows=[]
    assignment={}
    try:
        from scipy.optimize import linear_sum_assignment
        plan=output.plan[:-1,:-1].detach().float().cpu().numpy()
        finite=np.isfinite(plan)
        if finite.any():
            floor=float(plan[finite].min())-1e6
            plan=np.where(finite,plan,floor)
            rr,cc=linear_sum_assignment(-plan)
            assignment={int(r):int(c) for r,c in zip(rr,cc)}
    except ImportError:
        pass
    for identity,node in pairs:
        t.queries+=1; t.hungarian_queries+=1
        row=real[node]; row_all=probs[node]
        pred=int(row.argmax().item())
        dust=bool(row_all[-1]>row.max()); t.dustbin_top1+=int(dust)
        slot=identity_to_slot.get(identity)
        covered=slot is not None and 0<=int(slot)<real.shape[1]
        rank=None; correct=top5=top1db=hung=0; rrval=0.0; target_prob=None
        if covered:
            slot=int(slot); t.covered_queries+=1
            target=row[slot]
            rank=1+int((row>target).sum().item())
            partial=1+int((row_all>target).sum().item())
            correct=int(rank<=1); top5=int(rank<=min(5,real.shape[1]))
            rrval=1.0/rank; target_prob=float(target.item())
            top1db=int(partial<=1); hung=int(assignment.get(node,-1)==slot)
            t.top1+=correct; t.top5+=top5; t.reciprocal_rank_sum+=rrval
            t.top1_with_dustbin+=top1db; t.hungarian_correct+=hung
        rows.append({
            "query_worm":uid,"source_path":source_path,"node_index":node,"identity":identity,
            "candidate_covered":int(covered),"prediction_slot":pred,
            "prediction_identity":slot_to_identity.get(pred,f"__UNMAPPED_SLOT_{pred}"),
            "rank":"" if rank is None else rank,"correct":correct,"top5_correct":top5,
            "reciprocal_rank":rrval,"target_probability":"" if target_prob is None else target_prob,
            "dustbin_top1":int(dust),"top1_with_dustbin_correct":top1db,
            "hungarian_prediction_slot":assignment.get(node,-1),"hungarian_correct":hung,
        })
    return t,rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--package-root",type=Path,required=True); p.add_argument("--dataset-root",type=Path,required=True)
    p.add_argument("--split",choices=("val","test"),required=True); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--activity-length",type=int,default=512); p.add_argument("--device",default="cuda")
    p.add_argument("--dataset",required=True); p.add_argument("--fold",type=int,required=True)
    p.add_argument("--seed",type=int,required=True); p.add_argument("--variant",required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--query-output",type=Path,required=True)
    a=p.parse_args()
    import sys
    sys.path.insert(0,str(a.package_root.resolve()))
    from mprt_net.data import WormCache,split_files
    from mprt_net.evaluate import load_checkpoint
    device=torch.device(a.device if torch.cuda.is_available() else "cpu")
    model,ckpt=load_checkpoint(a.checkpoint,device); model.eval()
    mapping=ckpt.get("atlas_identity_to_slot")
    if not isinstance(mapping,dict) or not mapping: raise RuntimeError("missing atlas_identity_to_slot")
    identity_to_slot={str(k):int(v) for k,v in mapping.items()}
    slot_to_identity={v:k for k,v in identity_to_slot.items()}
    atlas=model.atlas_encoding()
    files=split_files(a.dataset_root,a.split)
    cache=WormCache(activity_length=a.activity_length,max_items=max(8,len(files)))
    total=Totals(); records=[]
    with torch.inference_mode():
        for n,path in enumerate(files,1):
            s_cpu=cache.get(path); s=s_cpu.to(device)
            out=model.match_encodings(model.encode_population(s),atlas)
            ct,cr=eval_sample(out,s,identity_to_slot,slot_to_identity,str(s_cpu.uid),str(s_cpu.source_path))
            total.update(ct); records.extend(cr)
            m=ct.metrics()
            print(f"evaluate {n:03d}/{len(files):03d} uid={s_cpu.uid} queries={ct.queries} covered={ct.covered_queries} coverage={100*m['candidate_coverage']:.2f}%",flush=True)
    result={
        "dataset":a.dataset,"dataset_root":str(a.dataset_root.resolve()),"split":a.split,
        "fold":a.fold,"seed":a.seed,"variant":a.variant,"checkpoint":str(a.checkpoint.resolve()),
        "checkpoint_epoch":ckpt.get("epoch"),"atlas_size":len(identity_to_slot),"recordings":len(files),
        "evaluation_protocol":"fixed_canonical_query_cohort_v1",
        "missing_gt_policy":"retain_as_incorrect",
        "query_definition":"all unique supervised identities in split, independent of atlas vocabulary",
        **total.metrics(),
    }
    write_json(a.output,result)
    a.query_output.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.query_output.with_suffix(a.query_output.suffix+".tmp")
    with tmp.open("w",encoding="utf-8",newline="") as f:
        if records:
            w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    tmp.replace(a.query_output)
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()

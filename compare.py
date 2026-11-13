#!/usr/bin/env python3
"""Compare independent E/F/V/stress evidence. Thresholds are frozen per gate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument("reference");p.add_argument("candidate");p.add_argument("--gate",choices=["compression","optimization"],required=True);p.add_argument("--output",required=True)
    a=p.parse_args()
    limits={"compression":{"energy_per_atom":1e-4,"force_rmse":1e-3,"force_max":5e-3,"virial_per_atom":1e-3,"stress_max":1e-4},
            "optimization":{"energy_per_atom":1e-6,"force_rmse":1e-5,"force_max":1e-4,"virial_per_atom":1e-5,"stress_max":1e-6}}[a.gate]
    refdir=Path(a.reference);canddir=Path(a.candidate)
    r=np.load(refdir/"outputs.npz");c=np.load(canddir/"outputs.npz")
    rm=json.loads((refdir/"result.json").read_text());cm=json.loads((canddir/"result.json").read_text())
    if rm.get("checkpoint_sha256")!=cm.get("checkpoint_sha256"):
        raise RuntimeError("original checkpoint identity mismatch")
    if set(r.files)!=set(c.files):raise RuntimeError("output cases/keys mismatch")
    cases={};passed=rm["status"]==cm["status"]=="PASS"
    for name in sorted(rm["cases"]):
        for key in ["positions","cell","numbers"]:
            if not np.array_equal(r[f"{name}/{key}"],c[f"{name}/{key}"]):raise RuntimeError("input mismatch "+name)
        n=rm["cases"][name]["natoms"]
        diffs={}
        for key in ["energy","forces","virial","stress"]:
            x=r[f"{name}/{key}"];y=c[f"{name}/{key}"]
            if x.shape!=y.shape or not np.isfinite(y).all():raise RuntimeError("shape/finite mismatch")
            diffs[key]=x-y
        metrics={"energy_per_atom":float(np.max(abs(diffs["energy"]))/n),"force_rmse":float(np.sqrt(np.mean(diffs["forces"]**2))),
            "force_max":float(np.max(abs(diffs["forces"]))),"virial_per_atom":float(np.max(abs(diffs["virial"]))/n),"stress_max":float(np.max(abs(diffs["stress"])))}
        elementwise={}
        if a.gate=='optimization':
            for key in ['energy','forces','virial','stress']:
                bound=1e-5+1e-4*np.abs(r[f'{name}/{key}'])
                elementwise[key]=float(np.max(np.abs(diffs[key])/bound))
        ok=all(metrics[k]<=v for k,v in limits.items()) and all(v<=1 for v in elementwise.values());passed &= ok
        cases[name]={"pass":ok,**metrics,"elementwise_error_over_tolerance":elementwise}
    if a.gate=="optimization" and rm.get("model_sha256")!=cm.get("model_sha256"):
        # Archives can differ due to codegen; original checkpoint identity must be
        # checked by the suite-level manifest, never inferred from artifact bytes.
        identity="archive SHA may differ; verify original checkpoint manifest"
    else:identity="reference and candidate outputs use identical fixed inputs"
    report={"status":"PASS" if passed else "FAIL","gate":a.gate,"limits":limits,"cases":cases,"identity_note":identity,
        "elementwise_optimization_rule":"abs(candidate-reference) <= 1e-5 + 1e-4*abs(reference)",
        "reference_result_sha256":hashlib.sha256((refdir/"result.json").read_bytes()).hexdigest(),"candidate_result_sha256":hashlib.sha256((canddir/"result.json").read_bytes()).hexdigest()}
    Path(a.output).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    if not passed:raise SystemExit(2)


if __name__=="__main__":main()

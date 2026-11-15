#!/usr/bin/env python3
"""Export without distributed communication artifact; compile time is separate."""
import argparse
import json
import time
import os
import hashlib
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True)
a=p.parse_args()
from deepmd.pt_expt.entrypoints.main import freeze
start=time.perf_counter()
freeze(a.checkpoint,a.output,head="Default",lower_kind="graph",with_comm=False,compress=True)
record={"status":"PASS","checkpoint":a.checkpoint,"archive":a.output,"export_seconds":time.perf_counter()-start,
        "checkpoint_sha256":hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        "archive_sha256":hashlib.sha256(Path(a.output).read_bytes()).hexdigest(),
        "export_flags":{k:v for k,v in os.environ.items() if k.startswith(("DP_","DPA4C_","CUDA_","NVIDIA_"))}}
Path(a.output+".export.json").write_text(json.dumps(record,indent=2))
print(json.dumps(record))

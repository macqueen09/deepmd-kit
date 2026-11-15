"""Pin one installed native library before the source-selection bootstrap."""

import hashlib
import os
import runpy
from pathlib import Path

import torch

native = Path(os.environ["DPA4C_NATIVE_LIBRARY"]).resolve()
expected = os.environ["DPA4C_NATIVE_SHA256"]
assert hashlib.sha256(native.read_bytes()).hexdigest() == expected
assert "libdeepmd_op_pt.so" not in Path("/proc/self/maps").read_text()
torch.ops.load_library(str(native))
try:
    runpy.run_path(str(Path(__file__).with_name("bootstrap.py")), run_name="__main__")
finally:
    actual = sorted({
        line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
        if "libdeepmd_op_pt.so" in line
    })
    assert actual == [str(native)]
    assert hashlib.sha256(native.read_bytes()).hexdigest() == expected

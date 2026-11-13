# C01 public reproduction and numerical evidence

This companion is separate from the five-file code patch. It contains no model binaries, credentials, private reports or machine-specific absolute paths. `results.json.gz` is a neutral, complete numerical attachment; `accuracy_tables.md` gives the actual primary-input three-column self-repeat table. The seven Nano qualification failures are retained. Do not describe the study as all-model strict PASS.

## Inputs and versions

- Tested BASE: `ba95134a1caa9fb658ccaa9f226ff54858f9246f`.
- Tested patch SHA-256: `00fbe6265f226f96dca1d31cab61c14608ff139e5fbcf44bd14cff7523594a29`; no candidate commit was created during measurement.
- Public models: [OutisLi/DPA4C-Paper](https://huggingface.co/OutisLi/DPA4C-Paper/tree/062fc79df75bcd38a97a9d9554a07cba7c89975b), pinned revision `062fc79df75bcd38a97a9d9554a07cba7c89975b`. File paths were verified against that revision.
- RTX 5090, driver 595.91.07; Python 3.11.12, PyTorch 2.11.0+cu128, Triton 3.6.0; ASE 3.29, Vesin/Vesin-torch 0.6.2.
- Both arms use the same original native operator. Its measured SHA is `6f8916e565b91612469bb09114e3ca17eb4a5438d130556daf2b88a2706adced`; native sources are unchanged by the patch. Your locally rebuilt binary may have a different hash; keep that binary identical between arms and record it.

```bash
set -euo pipefail
mkdir -p models archives
for MODEL in Nano Mini Neo Air Plus; do
  hf download OutisLi/DPA4C-Paper "DPA4C-${MODEL}-OMat24.pt" \
    --revision 062fc79df75bcd38a97a9d9554a07cba7c89975b --local-dir models
done
```

Checkpoint and measured archive hashes are in `results.json.gz` under `models.files`. Re-exported AOT artifacts can differ bytewise because of toolchain/code-generation details; compare the **same archive** between BASE and PR rather than independently exporting each arm.

## Exact export operation

`export.py` is the frozen parameterized export script used by the recorded export receipts. It calls:

```python
from deepmd.pt_expt.entrypoints.main import freeze
freeze(checkpoint, output, head="Default", lower_kind="graph",
       with_comm=False, compress=True)
```

This is the recorded Python entry point, not a guessed CLI spelling. Compile/export time is excluded from inference timing.

```bash
export DP_CUDA_INFER=2 DP_TRITON_INFER=2 DP_TF32_INFER=0 DP_AMP_INFER=0
export NVIDIA_TF32_OVERRIDE=0 DP_COMPILE_INFER=0 DP_ACT_INFER=0
python export.py --checkpoint models/DPA4C-Neo-OMat24.pt \
  --output archives/DPA4C-Neo-OMat24.pt2
```

Use the tested PyTorch/CUDA toolchain and an installation with the DPA4C native operator. The first Neo artifact in the recorded study required clearing the executable GNU_STACK flag of generated AOT shared objects; this changed loader metadata, not model weights. Subsequent exports used a local linker wrapper adding `-Wl,-z,noexecstack`. On systems enforcing non-executable stacks, handle that loader requirement explicitly and record any normalized archive hash; do not change weights or precision.

## Benchmark endpoint and deterministic input

`benchmark.py`, `compare.py` and `export.py` are verbatim copies of the frozen scripts used in the study, not newly optimized benchmark implementations. `bootstrap.py` is the original verified-import bootstrap: it selects the supplied source tree while preserving the same installed native extension. It requires the study's scikit-build editable installation layout; an incompatible environment fails rather than silently importing the wrong tree.

Primary geometry: periodic cubic carbon diamond, a=3.567 Å; repeat 5/8/10 → 1000/4096/8000 atoms; Gaussian jitter σ=0.02 Å, NumPy seed0. Auxiliary inputs are C64 seeds101/7/23, Cu108 FCC a=3.615 seed102, NaCl64 rocksalt a=5.64 seed103, translated C64 and rotated C64 (0.37 radians), all generated in the attached benchmark. No saved private input dataset is needed.

The endpoint explicitly calls `ASE.calculate(energy,forces,virial,stress)` every sample; it is not ASE result-cache lookup. Deployed arms use Vesin with skin1.0; ordinary `.pt` Torch references use cell neighbors and deterministic algorithms. All three retain original public output dtypes; strict FP32 describes compute, not an artificial conversion of all outputs.

## One measured arm

Install/build BASE once in a fresh editable environment. Prepare separate BASE and patched source trees at the tested base; preserve the same native installation. Set `SOURCE` to the appropriate tree for each fresh process. Paths below are user parameters, not paths from the measurement machine.

```bash
set -euo pipefail
export DP_CUDA_INFER=2 DP_TRITON_INFER=2 DP_TF32_INFER=0 DP_AMP_INFER=0
export NVIDIA_TF32_OVERRIDE=0 DP_COMPILE_INFER=0 DP_ACT_INFER=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DP_INTRA_OP_PARALLELISM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=1
unset LD_PRELOAD DP_DPA4C_COSCHEDULED_FACTOR CUBLAS_WORKSPACE_CONFIG
export DPA4C_EXPECTED_GPU_NAME="$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader)"
export DPA4C_EXPECTED_GPU_UUID="$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES="$DPA4C_EXPECTED_GPU_UUID"
export DEVICE=cuda LOCAL_RANK=0

# Set SOURCE to your BASE or patched source checkout; OUT must be a fresh directory.
test -n "$SOURCE"
test -n "$OUT"
mkdir -p "$OUT"
export DPA4C_SELECTED_SOURCE="$SOURCE"
export DPA4C_PROBE_PATH="$OUT/source_probe.json"
export TORCHINDUCTOR_CACHE_DIR="$OUT/cache/inductor"
export TRITON_CACHE_DIR="$OUT/cache/triton"
export CUDA_CACHE_PATH="$OUT/cache/cuda"
numactl --cpunodebind=0 --membind=0 python bootstrap.py benchmark.py \
  --model archives/DPA4C-Neo-OMat24.pt2 --checkpoint models/DPA4C-Neo-OMat24.pt \
  --expected-code-root "$SOURCE" --route deployed --neighbor vesin \
  --repeat 8 --warmup 100 --steps 500 --output "$OUT/evidence"
```

Check that no other compute process is active on the selected GPU before each arm; stop rather than kill it or change clocks. Verify `source_probe.json`, model hashes and the actual loaded native library in `result.json` for every process.

Repeat **six independent pairs**, orders AB, BA, AB, BA, AB, BA. Every arm gets a new process, output folder and cache. Tested cells: Nano/Mini/Air/Plus at N1000/4096, Neo at N1000/4096/8000 (66 pairs total). Use two independent ordinary-Torch correctness references per cell; do not replace a large reference with a smaller one.

For the matching **Neo4096** reference, select the BASE source as `SOURCE` and a fresh `REF_OUT`. Keep the GPU/strict settings above, but scope the reference-only CUDA/workspace settings to this child process so they do not leak into a later deployed arm:

```bash
test -n "$SOURCE"
test -n "$REF_OUT"
mkdir -p "$REF_OUT"
DP_CUDA_INFER=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
DPA4C_SELECTED_SOURCE="$SOURCE" DPA4C_PROBE_PATH="$REF_OUT/source_probe.json" \
TORCHINDUCTOR_CACHE_DIR="$REF_OUT/cache/inductor" \
TRITON_CACHE_DIR="$REF_OUT/cache/triton" CUDA_CACHE_PATH="$REF_OUT/cache/cuda" \
numactl --cpunodebind=0 --membind=0 python bootstrap.py benchmark.py \
  --model models/DPA4C-Neo-OMat24.pt --checkpoint models/DPA4C-Neo-OMat24.pt \
  --expected-code-root "$SOURCE" --route reference --neighbor cell \
  --repeat 8 --correctness-only --output "$REF_OUT/evidence"
```

Use repeat5 for N1000 or repeat10 for N8000 and the same model/input specification as the corresponding deployed arm. The two reference processes each retain five self-repeats for the actual `c64` and primary inputs. This study observed zero self-repeat differences in that deterministic reference configuration; it does not imply universal zero error or determinism for other configurations.

Compare each actual pair with the attached frozen gate:

```bash
python compare.py "$BASE_EVIDENCE" "$PR_EVIDENCE" \
  --gate optimization --output pair_optimization_gate.json
```

It retains failures. Separately compare each deployed arm against each matching-input Torch reference using `--gate compression`. Do not use compression PASS to override the stricter optimization gate.

## Timing, memory and precision definitions

- Per process: 100 warmups, 500 raw wall-clock frames; load/export/compilation/profiling excluded. Report P50/P90/min/max. Pair reduction = `100*(1-PR_i/BASE_i)`; aggregate the six percentages by median. The 95% bootstrap resamples independent pairs (10,000 draws, seed20261113), never individual frames.
- Record allocated/reserved separately. Including-first peak is the maximum of initialization/correctness, warmup and steady windows. In Mini4096, PR steady allocated is 66.385254 MiB while including-first is 66.635254 MiB; they must not be conflated.
- Five self-repeats after initial output, for actual `c64` and primary only. Torch has two reference processes, BASE/PR six each. Every eight-case cross-version comparison additionally checks all E/F/V/stress arrays and inputs.
- Strict absolute limits: E/atom1e-6, F-RMSE1e-5, Fmax1e-4, V/atom1e-5, stressmax1e-6; all elements also satisfy `abs(diff)<=1e-5+1e-4*abs(BASE)` and are finite. The original self-repeat gate uses absolute limits; extra elementwise self diagnostics are reported separately.
- All 88 predeclared calls completed. Nine non-Nano cells qualified 54/54 pairs; Nano qualified 5/12, with seven failures preserved. Nano diagnostic performance is not promoted as a qualified speedup. Controlled fixed-graph results do not erase those failures.

## Community regression tests

Run from the patched repository with its normal test dependencies:

```bash
python -m pytest source/tests/pt_expt/infer/test_request_source_csr.py \
  source/tests/pt_expt/utils/test_vesin_graph_builder.py \
  source/tests/pt_expt/infer/test_deep_eval_metadata_only.py \
  source/tests/pt_expt/infer/test_deep_eval_pt_checkpoint.py -q
python -m pytest source/tests/pt_expt/infer/test_request_source_csr_multinode_cuda.py -q
ruff check .
ruff format --check .
```

Final source regressions: 135 CPU passes, 24 skips, 45 subtests; six executed native CUDA passes; Neo/Plus twelve actual public-interface requests passed. CPU/GPU skips are not evidence of execution. The original installed native and relevant native source match the tested master; this study did not rebuild unrelated native changes.

## JSON and checksums

`results.json.gz` schema `dpa4c_c01_public_evidence_v1` includes every cell, pair, original 500-frame timing array, strict failure, cross/self error range, both Torch references' precision summaries, actual interface observations and fixed-graph diagnostic summaries. `evidence/...` strings are neutral evidence IDs, not paths on a private machine. No unpublished candidate commit SHA is invented.

`SHA256SUMS.json` covers the public companion files. The bundle is an optional attachment for manual review; it is not part of the minimal production patch.

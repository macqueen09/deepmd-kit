# DPA4C canonical graph preparation: reproduction companion

Submitted code commit: `97411854d643eea6095a1ad68e5518c56b5864be` in the `perf/dpa4c-geometry-lifetime-mask-safe-hun` branch. This evidence branch is separate from production code.

- [Accuracy ranges and retained failures](accuracy_tables.md): ordinary Torch / BASE / PR, all five metrics and all eleven model/size cells.
- [P90 and both reserved peaks](allocator_and_latency.md).
- [Full measured results](results.json.gz), [submission-day recheck](submission_recheck.json), [source/model/native identities](qualification_manifest.json), [checksums](SHA256SUMS).
- [Peak migration with C01 already applied](historical_C01_peak_migration.json), separately from [the independent-master cold/profile check](cold_and_profile_independent.json). These baselines must not be combined.

The original 46-pair matrix uses the final mask-safe implementation. A 2026-11-15 check adds two AB/BA pairs each for Neo4096 and Plus4096, all four passing the unchanged optimization gate; these do not replace the original matrix or its failures. Fresh CPU tests:146 PASS/29 CUDA SKIP/45 subtests; focused CUDA:125 PASS. Full original accuracy gate:41/46 PASS, not universal strict qualification.

The C01-stacked historical Plus4096 reserved peak increases136→144MiB. In the independent-master matrix, Neo8000 reserved increases166→168MiB. Allocated reductions do not imply every memory metric falls.

Paths in JSON are parameterized as `${REPRO_ROOT}`; the per-file hashes still refer to the original raw evidence. The local raw output archives are indexed by the manifest; this branch includes complete timing samples and numerical summaries, not the model binaries or all NPZ arrays. Regenerate arrays with the supplied scripts and public models.

This companion is separate from the six-file code patch. It contains no model binaries, credentials or private reports. The tested patch is independent of PR #6139: it retains the full source CSR.

## Sources and assets

- BASE: `ba95134a1caa9fb658ccaa9f226ff54858f9246f` (upstream master at qualification start).
- Candidate: the supplied `mask_safe_vs_master.patch`, applied to BASE; its SHA and three production-file hashes are in the qualification manifest.
- Public checkpoints: [OutisLi/DPA4C-Paper](https://huggingface.co/OutisLi/DPA4C-Paper/tree/062fc79df75bcd38a97a9d9554a07cba7c89975b), revision `062fc79df75bcd38a97a9d9554a07cba7c89975b`.
- Tested environment: RTX 5090, driver 595.91.07, Python 3.11.12, PyTorch 2.11.0+cu128, Triton 3.6.0. Both arms load the identical native operator and model archive.

```bash
for MODEL in Nano Mini Neo Air Plus; do
  hf download OutisLi/DPA4C-Paper "DPA4C-${MODEL}-OMat24.pt" \
    --revision 062fc79df75bcd38a97a9d9554a07cba7c89975b --local-dir models
done
```

Build/install BASE once with the DPA4C native CUDA operator. Prepare a separate patched source tree; do not rebuild a different native operator for the candidate. `bootstrap.py` verifies the selected Python tree while keeping the same scikit-build editable installation. Incompatible installation layouts fail explicitly. `guarded_bootstrap.py` additionally pins the installed native library by absolute path and SHA, matching the measured preload discipline.

`benchmark.py`, `compare.py`, and `export.py` are copies of the frozen measurement scripts. The explicit native guard and import bootstrap do not change model math.

## Export once, then share the archive

```bash
export DP_CUDA_INFER=2 DP_TRITON_INFER=2 DP_TF32_INFER=0 DP_AMP_INFER=0
export NVIDIA_TF32_OVERRIDE=0 DP_COMPILE_INFER=0 DP_ACT_INFER=0
python export.py --checkpoint models/DPA4C-Neo-OMat24.pt \
  --output archives/DPA4C-Neo-OMat24.pt2
```

The script calls `deepmd.pt_expt.entrypoints.main.freeze(..., head="Default", lower_kind="graph", with_comm=False, compress=True)`. Export/JIT time is not inference latency. Independently re-exported archives can differ because of code generation: use the same archive in both arms and record its SHA. On loaders that reject executable stacks, the recorded artifacts used a no-executable-stack linker setting / GNU_STACK metadata normalization; this did not change weights or arithmetic.

## Measured inputs and endpoint

Five checkpoints: Nano/Mini/Neo/Air/Plus. Each uses periodic carbon diamond, a=3.567 Å, repeat 5 or 8 → N1000/4096; Neo additionally repeat10 → N8000. All primary positions receive Gaussian jitter, standard deviation0.02 Å, NumPy seed0. The benchmark generates its own inputs, including C64 variants, Cu108, NaCl64, translation and rotation cases. No private input dataset is required.

Each sample explicitly invokes ASE `calculate(E,F,V,stress)`; it is not an ASE result-cache lookup. Vesin skin1.0 is reused. Neighbor preparation and host output copies are included. The formal matrix uses four independent AB/BA pairs per cell, six for Neo4096, for 46 pairs. Every arm has 100 warmups and 500 raw timed samples. Both steady and including-first allocator peaks are reported; the latter includes construction, correctness requests, warmup and timed inference. A separate fresh-process primary-first check verifies the cold-request behavior.

## One deployed arm

Set `SOURCE` to the chosen BASE or candidate checkout, `OUT` to a new output directory, and `DPA4C_NATIVE_LIBRARY` to the identical installed `libdeepmd_op_pt.so` used for both arms. Use the NUMA node attached to the selected card (GPU1/NUMA1 in the formal matrix).

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DP_INTRA_OP_PARALLELISM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=1
unset LD_PRELOAD DP_DPA4C_COSCHEDULED_FACTOR CUBLAS_WORKSPACE_CONFIG
export DPA4C_EXPECTED_GPU_NAME="$(nvidia-smi --id=1 --query-gpu=name --format=csv,noheader)"
export DPA4C_EXPECTED_GPU_UUID="$(nvidia-smi --id=1 --query-gpu=uuid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES="$DPA4C_EXPECTED_GPU_UUID" DEVICE=cuda LOCAL_RANK=0
export DPA4C_NATIVE_SHA256="$(sha256sum "$DPA4C_NATIVE_LIBRARY" | cut -d' ' -f1)"
mkdir -p "$OUT"
export DPA4C_SELECTED_SOURCE="$SOURCE" DPA4C_PROBE_PATH="$OUT/source_probe.json"
export TORCHINDUCTOR_CACHE_DIR="$OUT/cache/inductor" TRITON_CACHE_DIR="$OUT/cache/triton"
export CUDA_CACHE_PATH="$OUT/cache/cuda"
numactl --cpunodebind=1 --membind=1 python guarded_bootstrap.py benchmark.py \
  --model archives/DPA4C-Neo-OMat24.pt2 --checkpoint models/DPA4C-Neo-OMat24.pt \
  --expected-code-root "$SOURCE" --route deployed --neighbor vesin \
  --repeat 8 --warmup 100 --steps 500 --output "$OUT/evidence"
```

Check the selected GPU for other compute processes before each arm and monitor throughout the run. Stop rather than kill another user's process or modify clocks. The study records 0.5-second PID/clock/temperature/power samples; control failures are retained, excluded from performance, and rerun only with explicit new control-retry tags. Numerical failures are not rerun to select passing observations.

## Three-column accuracy

Run two ordinary-Torch references per model/size, with the original `.pt` checkpoint and fresh directories. Keep the same strict flags, but scope the following settings to the reference child:

```bash
DP_CUDA_INFER=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python guarded_bootstrap.py benchmark.py \
  --model models/DPA4C-Neo-OMat24.pt --checkpoint models/DPA4C-Neo-OMat24.pt \
  --expected-code-root "$SOURCE" --route reference --neighbor cell \
  --repeat 8 --correctness-only --output "$REF_OUT/evidence"
```

Use a fresh `OUT`, `REF_OUT`, import probe and cache for every child. Ordinary-Torch runs enable deterministic algorithms and are used for accuracy only, never for cross-device performance comparison. Each arm records five self repeats on C64 and the primary input. Compare:

```bash
python compare.py "$BASE_OUT/evidence" "$PR_OUT/evidence" \
  --gate optimization --output paired_gate.json
python compare.py "$REF_OUT/evidence" "$BASE_OUT/evidence" \
  --gate compression --output base_vs_torch.json
python compare.py "$REF_OUT/evidence" "$PR_OUT/evidence" \
  --gate compression --output pr_vs_torch.json
```

The optimization gate retains the original aggregate E/F/V/stress thresholds and elementwise `abs(PR-BASE) <= 1e-5 + 1e-4*abs(BASE)`. Compression tolerances are separately defined in `compare.py`, not substituted for the optimization gate. Match actual checkpoint hashes and every case's input shape/dtype/bytes. Keep self-repeat flags, case-level failures and overall status separately: an inherited BASE self-repeat failure can make an overall comparison fail even when its cross-implementation case metrics pass.

The complete qualification retains these failures rather than claiming universal strict PASS. Fixed-graph bitwise comparisons are auxiliary evidence; they do not overwrite default-pipeline failures.

## Masked and DPA4 checks

The final implementation narrows geometry **after node masking, before concatenation/sorting**. This avoids the first prototype's wasted conversion of discarded edges. Mask fractions0%,1%,50%,100% were checked separately; local graph peaks must not be mislabeled as full-model peaks.

Additional checks cover dynamic positions/cells/types, open/periodic boundaries, multi-frame inputs, normal→atomic→normal requests, graph-field equality and integer-width boundaries. Existing DPA4 `.pt` and ordinary graph `.pt2` paths are tested separately; their pre-existing master test failures are listed rather than hidden. None of these tests changes model weights, the native library or floating-point tolerances.

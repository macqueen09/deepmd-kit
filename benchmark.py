#!/usr/bin/env python3
"""Frozen-geometry, cache-safe ASE E/F/V/stress baseline and regression runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=30)
    return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", required=True, help="Original immutable checkpoint for model identity")
    p.add_argument("--output", required=True)
    p.add_argument("--expected-code-root", required=True)
    p.add_argument("--route", choices=["reference", "deployed"], required=True)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--correctness-only", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--neighbor", choices=["cell","vesin","auto"], default="vesin")
    args = p.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    model = Path(args.model).resolve()
    result = {"schema": "dpa4c_rtx5090_acceptance_v1", "status": "STARTED", "args": vars(args),
              "evidence_class": "SHARED_GPU_DIRECTIONAL", "model": str(model), "model_sha256": digest(model),
              "checkpoint_sha256": digest(args.checkpoint),
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hostname": platform.node()}
    result["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    try:
        import numpy as np
        import torch
        import deepmd
        from ase.build import bulk
        from deepmd.calculator import DP
        code_root = Path(args.expected_code_root).resolve()
        if not Path(deepmd.__file__).resolve().is_relative_to(code_root):
            raise RuntimeError(f"wrong code: {deepmd.__file__}, expected {code_root}")
        if os.environ.get("DP_TRITON_INFER") != "2":
            raise RuntimeError("DP_TRITON_INFER must be 2")
        if os.environ.get("DP_TF32_INFER") != "0" or os.environ.get("DP_AMP_INFER") != "0":
            raise RuntimeError("strict FP32 flags required")
        if args.route == "reference" and model.suffix != ".pt":
            raise RuntimeError("reference must be original .pt checkpoint")
        if args.route == "deployed" and model.suffix != ".pt2":
            raise RuntimeError("deployed must be frozen .pt2 archive")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        if args.route == "reference":
            torch.use_deterministic_algorithms(True)
        torch.cuda.reset_peak_memory_stats()
        expected_name = os.environ["DPA4C_EXPECTED_GPU_NAME"]
        if (
            "5090" not in expected_name
            or torch.cuda.get_device_name() != expected_name
            or torch.cuda.get_device_capability() != (12, 0)
        ):
            raise RuntimeError("RTX 5090 device identity mismatch")
        selected=command(["nvidia-smi","-i",os.environ["CUDA_VISIBLE_DEVICES"],"--query-gpu=uuid","--format=csv,noheader"])
        if selected["returncode"] or selected["stdout"]!=os.environ.get("DPA4C_EXPECTED_GPU_UUID"):
            raise RuntimeError("GPU UUID mismatch: "+str(selected))
        result["gpu_uuid"]=selected["stdout"]
        result["environment"] = {"python": sys.version, "executable": sys.executable,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "deepmd": deepmd.__version__, "deepmd_path": deepmd.__file__,
            "gpu_name": torch.cuda.get_device_name(), "gpu_cc": torch.cuda.get_device_capability(),
            "flags": {k:v for k,v in os.environ.items() if k.startswith(("DP_", "CUDA_", "TORCH", "OMP_", "NVIDIA_"))}}
        result["gpu_before"] = command(["nvidia-smi", "--query-gpu=index,name,uuid,memory.used,utilization.gpu,temperature.gpu,pstate,clocks.current.sm,clocks.current.memory,power.draw", "--format=csv"])
        source_hash = hashlib.sha256()
        for folder, extensions in [(code_root/"deepmd", {".py"}), (code_root/"source/op/pt", {".cu", ".cuh", ".cc", ".h", ".in", ".txt"})]:
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix in extensions and "__pycache__" not in str(path):
                    source_hash.update(str(path.relative_to(code_root)).encode())
                    source_hash.update(bytes.fromhex(digest(path)))
        result["source_tree_sha256"] = source_hash.hexdigest()
        t0=time.perf_counter()
        calc = DP(model=str(model), freeze=False, neighbor_graph_method=args.neighbor, neighbor_skin=1.0)
        call_counter=[0]
        original_eval=calc.dp.eval
        def counted_eval(*positional,**keywords):
            call_counter[0]+=1
            return original_eval(*positional,**keywords)
        calc.dp.eval=counted_eval
        result["load_seconds"] = time.perf_counter()-t0
        evaluator = calc.dp.deep_eval
        result["metadata"] = getattr(evaluator, "metadata", {})
        if args.route == "deployed" and result["metadata"].get("lower_input_kind") != "dpa4c_canonical":
            raise RuntimeError("deployed baseline must use the DPA4C compact canonical ABI")
        result["type_map"] = calc.dp.get_type_map()
        arrays = {}
        cases = {}

        def make(element, crystal, lattice, repeat, seed):
            atoms = bulk(element, crystal, a=lattice, cubic=True).repeat((repeat,)*3)
            atoms.positions += np.random.default_rng(seed).normal(0, .02, atoms.positions.shape)
            return atoms

        c64=make("C", "diamond", 3.567, 2, 101)
        cu=make("Cu", "fcc", 3.615, 3, 102)
        nacl=bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2,2,2))
        nacl.positions += np.random.default_rng(103).normal(0,.02,nacl.positions.shape)
        primary=make("C", "diamond",3.567,args.repeat,0)
        cases.update(c64=c64, cu108=cu, nacl64=nacl, primary=primary)
        for seed in (7,23):
            cases[f"c64_seed{seed}"]=make("C","diamond",3.567,2,seed)
        translated=c64.copy(); translated.positions += [0.173,-0.219,0.137]
        angle=.37
        rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        rotated=c64.copy();rotated.positions=c64.positions@rotation.T;rotated.set_cell(c64.cell.array@rotation.T,scale_atoms=False)
        cases.update(translated=translated,rotated=rotated)

        def evaluate(atoms):
            # Calling calculate forces an evaluation even when atoms did not change.
            calc.calculate(atoms, properties=["energy","forces","virial","stress"])
            return {k:np.asarray(calc.results[k]).copy() for k in ("energy","forces","virial","stress")}

        result["cases"]={}
        for name,atoms in cases.items():
            values=evaluate(atoms)
            for key,value in values.items():
                if not np.isfinite(value).all(): raise RuntimeError(f"nonfinite {name}/{key}")
                arrays[f"{name}/{key}"]=value
            arrays[f"{name}/positions"]=atoms.positions.copy()
            arrays[f"{name}/cell"]=atoms.cell.array.copy()
            arrays[f"{name}/numbers"]=atoms.numbers.copy()
            result["cases"][name]={"natoms":len(atoms),"positions_sha256":hashlib.sha256(atoms.positions.tobytes()).hexdigest(),
                "energy":float(values["energy"]),"force_rms":float(np.sqrt(np.mean(values["forces"]**2)))}
        result["symmetry"]={
            "translation_energy_per_atom":float(abs(arrays["c64/energy"]-arrays["translated/energy"])/len(c64)),
            "translation_force_max":float(np.max(abs(arrays["c64/forces"]-arrays["translated/forces"]))),
            "rotation_energy_per_atom":float(abs(arrays["c64/energy"]-arrays["rotated/energy"])/len(c64)),
            "rotation_force_max":float(np.max(abs(arrays["c64/forces"]@rotation.T-arrays["rotated/forces"])))}
        # Sanity check only: finite-difference FP32 energies are cancellation-sensitive.
        fd=[]
        for h in (.01,.02):
            plus=c64.copy();minus=c64.copy();plus.positions[0,0]+=h;minus.positions[0,0]-=h
            ep=float(evaluate(plus)["energy"]);em=float(evaluate(minus)["energy"])
            numeric=-(ep-em)/(2*h);analytic=float(arrays["c64/forces"][0,0])
            fd.append({"h_A":h,"numeric_force":numeric,"analytic_force":analytic,"abs_error":abs(numeric-analytic)})
        result["finite_difference_sanity"]=fd
        result["numerical_sanity_pass"] = all(result["symmetry"][k] <= (1e-4 if "energy" in k else 5e-3) for k in result["symmetry"]) and min(x["abs_error"] for x in fd)<.03
        np.savez_compressed(out/"outputs.npz",**arrays)
        # Same-model repeatability is checked independently from candidate parity.
        noise={};noise_arrays={}
        for name in ["c64","primary"]:
            samples=[]
            for repeat_index in range(5):
                vals=evaluate(cases[name]);n=len(cases[name])
                for key,value in vals.items():noise_arrays[f"{name}/{repeat_index}/{key}"]=value
                de=vals["energy"]-arrays[f"{name}/energy"]
                df=vals["forces"]-arrays[f"{name}/forces"]
                dv=vals["virial"]-arrays[f"{name}/virial"]
                ds=vals["stress"]-arrays[f"{name}/stress"]
                samples.append({"energy_per_atom":float(np.max(abs(de))/n),"force_rmse":float(np.sqrt(np.mean(df**2))),"force_max":float(np.max(abs(df))),"virial_per_atom":float(np.max(abs(dv))/n),"stress_max":float(np.max(abs(ds)))})
            noise[name]=samples
        result["self_repeatability"]=noise
        repeat_limits={"energy_per_atom":1e-6,"force_rmse":1e-5,"force_max":1e-4,"virial_per_atom":1e-5,"stress_max":1e-6}
        result["self_repeatability_limits"]=repeat_limits
        result["self_repeatability_pass"]=all(sample[key]<=limit for samples in noise.values() for sample in samples for key,limit in repeat_limits.items())
        np.savez_compressed(out/"self_repeat_outputs.npz",**noise_arrays)
        result["initialization_and_correctness_peak_allocated_MiB"]=torch.cuda.max_memory_allocated()/2**20
        result["initialization_and_correctness_peak_reserved_MiB"]=torch.cuda.max_memory_reserved()/2**20

        if not args.correctness_only:
            result["first_call_is_excluded"] = True
            for _ in range(args.warmup): evaluate(primary)
            torch.cuda.synchronize()
            result["pre_timing_peak_allocated_MiB"]=torch.cuda.max_memory_allocated()/2**20
            result["pre_timing_peak_reserved_MiB"]=torch.cuda.max_memory_reserved()/2**20
            torch.cuda.reset_peak_memory_stats()
            mem_before=torch.cuda.memory_stats()
            latencies=[]
            eval_before=call_counter[0]
            for _ in range(args.steps):
                torch.cuda.synchronize();start=time.perf_counter()
                evaluate(primary)
                torch.cuda.synchronize();latencies.append((time.perf_counter()-start)*1000)
            stats=torch.cuda.memory_stats()
            x=np.array(latencies)
            result["performance"]={"endpoint":"ASE calculate: E+F+V+stress, includes neighbor preparation and host output copies; fixed geometry, skin reuse",
                "natoms":len(primary),"warmup":args.warmup,"steps":args.steps,"mean_ms":float(x.mean()),"p50_ms":float(np.median(x)),"p90_ms":float(np.percentile(x,90)),
                "min_ms":float(x.min()),"max_ms":float(x.max()),
                "cv_percent":float(x.std()/x.mean()*100),"atoms_per_ms":float(len(primary)/x.mean()),"latencies_ms":latencies,
                "peak_allocated_MiB":torch.cuda.max_memory_allocated()/2**20,"peak_reserved_MiB":torch.cuda.max_memory_reserved()/2**20,
                "allocator_retry_delta":stats.get("num_alloc_retries",0)-mem_before.get("num_alloc_retries",0),
                "oom_delta":stats.get("num_ooms",0)-mem_before.get("num_ooms",0),
                "timed_model_eval_calls":call_counter[0]-eval_before}
            result["including_first_execution_peak_allocated_MiB"]=max(result["pre_timing_peak_allocated_MiB"],result["performance"]["peak_allocated_MiB"])
            result["including_first_execution_peak_reserved_MiB"]=max(result["pre_timing_peak_reserved_MiB"],result["performance"]["peak_reserved_MiB"])
            if call_counter[0]-eval_before != args.steps:raise RuntimeError("timed model call count mismatch")
        if args.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
                for _ in range(5): evaluate(primary)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(out/"profile.json"))
            (out/"profile_table.txt").write_text(prof.key_averages().table(sort_by="self_cuda_time_total",row_limit=60))
        result["loaded_libraries"]={s:digest(s) for s in sorted(torch.ops.loaded_libraries) if Path(s).is_file()}
        result["gpu_after"]=command(["nvidia-smi","--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu,pstate,clocks.current.sm,power.draw","--format=csv"])
        result["status"]="PASS" if result["numerical_sanity_pass"] and result["self_repeatability_pass"] else "SANITY_OR_REPEATABILITY_FAIL"
    except Exception as exc:
        import traceback
        result.update(status="ERROR",error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        (out/"result.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    print(json.dumps({"status":result["status"],"output":str(out),"performance":{k:v for k,v in result.get("performance",{}).items() if k!="latencies_ms"}},default=str))


if __name__ == "__main__":
    main()

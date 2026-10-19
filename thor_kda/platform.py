"""Thor identity and write-boundary checks for KDA kernel candidates."""
from __future__ import annotations
import hashlib, importlib.metadata, json, os, platform as host, subprocess, sys
from pathlib import Path

ROOT = Path(os.environ.get("KDA_THOR_ROOT", Path(__file__).resolve().parents[1]))
CUDA = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
PYTHON = Path(sys.executable)

def _run(*argv: str) -> dict:
    try:
        p = subprocess.run(argv, text=True, capture_output=True, timeout=30)
        return {"argv": list(argv), "returncode": p.returncode,
                "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"argv": list(argv), "returncode": -1, "stdout": "", "stderr": repr(exc)}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def discover() -> dict:
    gpu = _run("nvidia-smi", "--query-gpu=uuid,name,driver_version,pci.bus_id",
               "--format=csv,noheader")
    cc = _run("nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader")
    procs = _run("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                 "--format=csv,noheader")
    nvcc = _run(str(CUDA / "bin/nvcc"), "--version")
    ptxas = _run(str(CUDA / "bin/ptxas"), "--version")
    ncu = _run(str(CUDA / "bin/ncu"), "--version")
    power_mode = _run("nvpmodel", "-q")
    devfreq = {}
    for name, path in {
        "gpu_cur_hz": "/sys/class/devfreq/gpu-gpc-0/cur_freq",
        "gpu_target_hz": "/sys/class/devfreq/gpu-gpc-0/target_freq",
        "gpu_min_hz": "/sys/class/devfreq/gpu-gpc-0/min_freq",
        "gpu_max_hz": "/sys/class/devfreq/gpu-gpc-0/max_freq",
        "emc_cur_hz": "/sys/class/devfreq/bwmgr/cur_freq",
        "emc_target_hz": "/sys/class/devfreq/bwmgr/target_freq",
        "emc_min_hz": "/sys/class/devfreq/bwmgr/min_freq",
        "emc_max_hz": "/sys/class/devfreq/bwmgr/max_freq",
        "gpu_governor": "/sys/class/devfreq/gpu-gpc-0/governor",
        "emc_governor": "/sys/class/devfreq/bwmgr/governor",
    }.items():
        try: devfreq[name] = Path(path).read_text().strip()
        except OSError: devfreq[name] = None
    packages = {}
    for name in ("torch", "triton", "flashinfer-python", "nvidia-cutlass-dsl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    out = {
        "schema_version": 2, "root": str(ROOT), "python": sys.version,
        "python_executable": str(PYTHON), "host": {"hostname": host.node(), "arch": host.machine()},
        "gpu": {"identity": gpu, "compute_capability": cc, "active_processes": procs},
        "cuda": {"home": str(CUDA), "nvcc": nvcc, "ptxas": ptxas, "ncu": ncu},
        "power": {"mode": power_mode, "devfreq": devfreq,
                  "clock_note": "Thor nvidia-smi reports SM/EMC clocks as N/A; devfreq target/current values are diagnostics and are not substituted for measured clocks."},
        "packages": packages,
        "target": {"arch": "sm_110a", "cc_major": 11, "cc_minor": 0},
        "paths": {"cutlass": os.environ.get("CUTLASS_ROOT", ""),
                  "dsl_cache": str(ROOT / "cache/cute-dsl"),
                  "build": str(ROOT / "build")},
        "environment": {k: os.environ.get(k) for k in
                        ("CUDA_VISIBLE_DEVICES", "CUTE_DSL_ARCH", "TRITON_PTXAS_PATH")},
    }
    return out

def assert_root(path: Path) -> None:
    root = ROOT.resolve()
    resolved = path.resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError(f"write escapes task root: {resolved} not below {root}")

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(); assert_root(a.output); a.output.parent.mkdir(parents=True, exist_ok=True)
    record = discover(); a.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()

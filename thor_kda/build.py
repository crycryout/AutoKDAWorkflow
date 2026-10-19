"""Fixed CUTLASS and CuTe DSL build entry points."""
from __future__ import annotations
import json, os, shlex, subprocess
from pathlib import Path
from .platform import ROOT, CUDA, assert_root, sha256
from .ledger import record, artifact

def cutlass_command(source: Path, output: Path, cutlass: Path, defines: dict) -> list[str]:
    assert_root(output); source = source.resolve(); cutlass = cutlass.resolve()
    cmd = [str(CUDA / "bin/nvcc"), "-O3", "-std=c++17", "--expt-relaxed-constexpr",
           "-shared", "-Xcompiler", "-fPIC", "-lineinfo", "-Xptxas=-v",
           "-diag-suppress=20012", "-arch=sm_110a", "-I", str(cutlass / "include"),
           "-I", str(cutlass / "tools/util/include")]
    cmd.extend(f"-D{key}={val}" for key, val in sorted(defines.items()))
    cmd.extend(["-o", str(output), str(source)])
    return cmd

def build_cutlass(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    source = (ROOT / manifest["source"]).resolve(); output = (ROOT / manifest["output"]).resolve()
    cutlass_root = manifest.get("cutlass_root") or os.environ.get("CUTLASS_ROOT")
    if not cutlass_root:
        raise ValueError("Set CUTLASS_ROOT or manifest cutlass_root")
    cutlass = Path(cutlass_root).expanduser().resolve(); assert_root(source); assert_root(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = cutlass_command(source, output, cutlass, manifest.get("defines", {}))
    env = dict(os.environ); env.update({"CUDA_CACHE_PATH": str(ROOT / "cache/cuda"),
        "TMPDIR": str(ROOT / "tmp")})
    (ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    p = subprocess.run(cmd, text=True, capture_output=True, env=env)
    log = output.with_suffix(".build.log"); log.write_text(p.stdout + "\n" + p.stderr)
    result = {"candidate": manifest["name"], "command": cmd, "command_shell": shlex.join(cmd),
              "returncode": p.returncode, "log": artifact(log), "source": artifact(source)}
    if p.returncode == 0: result["binary"] = artifact(output)
    record("build", result)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / f"{manifest['name']}-build.json").write_text(json.dumps(result, indent=2) + "\n")
    if p.returncode: raise RuntimeError(f"CUTLASS build failed; see {log}")
    return result

def dsl_environment() -> dict:
    env = dict(os.environ); env.update({"CUTE_DSL_ARCH": "sm_110a",
        "CUTE_DSL_DUMP_DIR": str(ROOT / "results/cute-dsl-dump"),
        "CUTE_DSL_CACHE_DIR": str(ROOT / "cache/cute-dsl"),
        "CUDA_CACHE_PATH": str(ROOT / "cache/cuda"), "TMPDIR": str(ROOT / "tmp")})
    for p in (ROOT / "results/cute-dsl-dump", ROOT / "cache/cute-dsl", ROOT / "cache/cuda", ROOT / "tmp"): p.mkdir(parents=True, exist_ok=True)
    return env

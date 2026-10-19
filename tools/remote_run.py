#!/usr/bin/env python3
"""Run one Thor command with bounded writes and task-local measurement evidence.

The local controller supplies THOR_KDA_* runtime settings. On Thor itself, export
the same settings before running this script directly; SSH is not required.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thor_kda.config import RuntimeConfig


def runtime_config(config=None):
    return config if config is not None else RuntimeConfig.from_environ()


def run_read(argv, env=None):
    try:
        process = subprocess.run([str(x) for x in argv], text=True, capture_output=True, timeout=20, env=env)
        return dict(argv=[str(x) for x in argv], returncode=process.returncode,
                    stdout=process.stdout.strip(), stderr=process.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(argv=[str(x) for x in argv], returncode=-1, stdout="", stderr=str(error))


def bounded_root(path, config=None):
    config = runtime_config(config)
    if config.allowed_root is None:
        raise ValueError("Set THOR_KDA_ALLOWED_ROOT to the permitted workspace parent")
    path, allowed = Path(path).resolve(), config.allowed_root.resolve()
    if allowed == Path("/") or not path.is_relative_to(allowed) or path == allowed:
        raise ValueError("Thor writes must stay in a task directory strictly below allowed_root")
    return path


def snapshot():
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == "kB":
            memory[parts[0].rstrip(":")] = int(parts[1])
    clocks = {}
    for label, path in (("gpu", "/sys/class/devfreq/gpu-gpc-0/cur_freq"),
                        ("emc", "/sys/class/devfreq/bwmgr/cur_freq")):
        try:
            clocks[label] = Path(path).read_text().strip()
        except OSError:
            clocks[label] = None
    return dict(
        utc=datetime.now(timezone.utc).isoformat(), hostname=platform.node(), architecture=platform.machine(),
        gpu=run_read(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"]),
        gpu_processes=run_read(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]),
        meminfo_kib=memory, clocks_hz=clocks,
    )


def profiler_command(config, argv):
    return (["sudo", "-n", "/usr/bin/env"] if config.profiler_sudo else ["/usr/bin/env"]) + [str(x) for x in argv]


def doctor(config=None):
    config = runtime_config(config)
    out = snapshot()
    out["cuda"] = run_read([config.cuda_home / "bin/nvcc", "--version"])
    out["sudo"] = run_read(["sudo", "-n", "/usr/bin/id", "-u"]) if config.profiler_sudo else {"required": False}
    out["ncu"] = run_read(profiler_command(config, [config.ncu_executable, "--version"]))
    out["nsys"] = run_read([config.nsys, "--version"])
    out["packages"] = {}
    for name in ("torch", "triton", "nvidia-cutlass-dsl", "cuda-python", "apache-tvm-ffi"):
        try:
            out["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out["packages"][name] = None
    out["profiler_capture_verified"] = False
    return out


def gpu_pids(state):
    result = state["gpu_processes"]
    if result["returncode"] != 0:
        raise RuntimeError("Cannot establish GPU occupancy: " + result.get("stderr", ""))
    return [int(line.split(",")[0]) for line in result["stdout"].splitlines() if line.strip()]


def descendants(root):
    parents = {}
    for path in Path("/proc").glob("[0-9]*/status"):
        try:
            parent = int(re.search(r"^PPid:\s+(\d+)", path.read_text(), re.M).group(1))
            parents[int(path.parent.name)] = parent
        except (OSError, AttributeError, ValueError):
            continue
    found = {root}
    while True:
        more = {pid for pid, parent in parents.items() if parent in found} - found
        if not more:
            return found
        found |= more


def foreign_gpu_pids(state, owned_before, owned_after):
    return sorted(set(gpu_pids(state)) - (set(owned_before) | set(owned_after)))


def environment(root, storage_root=None, config=None):
    config = runtime_config(config)
    root = bounded_root(root, config)
    storage = bounded_root(storage_root if storage_root is not None else root, config)
    if not storage.is_relative_to(root):
        raise ValueError("Cache storage must remain inside this task")
    directories = {
        "TMPDIR": "tmp", "XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config",
        "HF_HOME": "cache/huggingface", "TORCH_EXTENSIONS_DIR": "build/torch_extensions",
        "TRITON_CACHE_DIR": "cache/triton", "CUDA_CACHE_PATH": "cache/cuda",
        "CUTE_DSL_CACHE_DIR": "cache/cute-dsl", "CUTE_DSL_DUMP_DIR": "results/cute-dsl-dump",
        "UV_CACHE_DIR": "cache/uv", "PIP_CACHE_DIR": "cache/pip",
    }
    additions = {key: str(bounded_root(storage / path, config)) for key, path in directories.items()}
    for value in additions.values():
        Path(value).mkdir(parents=True, exist_ok=True)
    additions.update(CUDA_HOME=str(config.cuda_home), CUTE_DSL_ARCH=config.cute_arch,
                     KDA_THOR_ROOT=str(root), PYTHONDONTWRITEBYTECODE="1",
                     NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT="1")
    env = dict(os.environ)
    env.update(additions)
    env["PATH"] = str(config.cuda_home / "bin") + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(root) + os.pathsep + str(Path(__file__).resolve().parents[1])
    return env, additions


def verify_identity(config, state):
    if platform.machine() != config.expected_arch:
        raise RuntimeError("Unexpected CPU architecture; expected " + config.expected_arch)
    if config.expected_hostname and platform.node() != config.expected_hostname:
        raise RuntimeError("Unexpected Thor hostname")
    if state["gpu"]["returncode"] or config.expected_gpu not in state["gpu"]["stdout"]:
        raise RuntimeError("Expected GPU identity: " + config.expected_gpu)


def execute(root, tag, argv, profile="none", cpu_only=False, ncu_kernel=None, config=None):
    config = runtime_config(config)
    root = bounded_root(root, config)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise ValueError("Invalid run tag")
    if profile not in ("none", "ncu", "nsys"):
        raise ValueError("Unknown profiler")
    if cpu_only and profile != "none":
        raise ValueError("CPU-only runs cannot collect GPU profiles")
    argv = [str(x).replace("{python}", config.python).replace("{root}", str(root)) for x in argv]
    if not argv:
        raise ValueError("A command after -- is required")
    before = snapshot()
    verify_identity(config, before)
    if not cpu_only and gpu_pids(before):
        raise RuntimeError("GPU busy; execution deferred. Existing compute PIDs: " + str(gpu_pids(before)))
    run_dir = bounded_root(root / "results" / tag, config)
    run_dir.mkdir(parents=True, exist_ok=False)
    env, additions = environment(root, run_dir / "scratch" if profile != "none" else None, config)
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    if profile == "ncu":
        ncu_options = ["--section-folder", config.ncu_sections] if config.ncu_sections else []
        command = profiler_command(config, [
            *(f"{key}={value}" for key, value in additions.items()),
            f"PATH={env['PATH']}", f"PYTHONPATH={env['PYTHONPATH']}", config.ncu_executable,
            *ncu_options, "--clock-control", "none", "--cache-control", "none", "--set", "basic",
            "--launch-count", "1", "--target-processes", "all", "--export", str(run_dir / "report"),
            *(["--kernel-name", ncu_kernel] if ncu_kernel else []), "--", *argv,
        ])
    elif profile == "nsys":
        command = [config.nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cuda-graph-trace=node",
                   "--output", str(run_dir / "report"), *argv]
    else:
        command = argv
    record = dict(command=command, cwd=str(root), environment=additions, before=before,
                  profile=profile, cpu_only=cpu_only, telemetry=[], returncode=None)
    metadata = run_dir / "run.json"
    metadata.write_text(json.dumps(record, indent=2))
    stop = threading.Event()

    def monitor():
        while not stop.wait(2):
            owned_before = descendants(os.getpid())
            try:
                state = snapshot()
                owned_after = descendants(os.getpid())
                state["owned_pids_before_snapshot"] = sorted(owned_before)
                state["owned_pids_after_snapshot"] = sorted(owned_after)
                state["foreign_gpu_pids"] = foreign_gpu_pids(state, owned_before, owned_after)
            except (OSError, RuntimeError, ValueError) as error:
                state = {"monitor_error": str(error)}
            record["telemetry"].append(state)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        with (run_dir / "stdout.log").open("w") as out, (run_dir / "stderr.log").open("w") as err:
            process = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err)
        record["returncode"] = process.returncode
    except OSError as error:
        record["execution_error"] = str(error)
        record["returncode"] = 127
    finally:
        stop.set()
        thread.join(timeout=50)
        record["interference_observed"] = any(
            state.get("foreign_gpu_pids") or state.get("monitor_error") for state in record["telemetry"]
        )
        try:
            record["after"] = snapshot()
            if not cpu_only:
                record["interference_observed"] |= bool(gpu_pids(record["after"]))
        except (OSError, RuntimeError, ValueError) as error:
            record["after_error"] = str(error)
            record["interference_observed"] = True
        metadata.write_text(json.dumps(record, indent=2))
    if profile == "ncu":
        report = run_dir / "report.ncu-rep"
        record["ncu_report_exists"] = report.is_file() and report.stat().st_size > 0
        if record["ncu_report_exists"]:
            imported = run_read(profiler_command(config, [
                *(f"{key}={value}" for key, value in additions.items()),
                config.ncu_executable, "--import", report, "--page", "details",
            ]), env=env)
            (run_dir / "import.stdout.log").write_text(imported.pop("stdout"))
            (run_dir / "import.stderr.log").write_text(imported.pop("stderr"))
            record["ncu_import"] = imported
            record["ncu_report_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
        record["ncu_capture_import_passed"] = (
            record["returncode"] == 0 and record.get("ncu_import", {}).get("returncode") == 0
            and (run_dir / "import.stdout.log").is_file() and (run_dir / "import.stdout.log").stat().st_size > 0
        )
    elif profile == "nsys":
        report = run_dir / "report.nsys-rep"
        record["nsys_report_exists"] = report.is_file() and report.stat().st_size > 0
        if record["returncode"] == 0 and record["nsys_report_exists"]:
            exported = run_read([config.nsys, "export", "--type", "sqlite", "--output",
                                 run_dir / "report.sqlite", report], env=env)
            record["nsys_export"] = exported
        record["nsys_capture_export_passed"] = (
            record.get("nsys_export", {}).get("returncode") == 0
            and (run_dir / "report.sqlite").is_file() and (run_dir / "report.sqlite").stat().st_size > 0
        )
    metadata.write_text(json.dumps(record, indent=2))
    return record, run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--task-root", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--profile", choices=("none", "ncu", "nsys"), default="none")
    parser.add_argument("--ncu-kernel", help="NCU kernel-name expression")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.doctor:
        print(json.dumps(doctor(), indent=2))
        return 0
    if not args.task_root or not args.tag:
        parser.error("--task-root and --tag are required")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    record, path = execute(args.task_root, args.tag, command, args.profile, args.cpu_only, args.ncu_kernel)
    print(json.dumps(dict(run_dir=str(path), returncode=record["returncode"],
                          interference_observed=record["interference_observed"],
                          ncu_capture_import_passed=record.get("ncu_capture_import_passed"),
                          nsys_capture_export_passed=record.get("nsys_capture_export_passed")), indent=2))
    return (record["returncode"] or (3 if record["interference_observed"] and not args.cpu_only else 0)
            or (4 if args.profile == "ncu" and not record.get("ncu_capture_import_passed") else 0)
            or (5 if args.profile == "nsys" and not record.get("nsys_capture_export_passed") else 0))


if __name__ == "__main__":
    raise SystemExit(main())

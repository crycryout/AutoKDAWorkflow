#!/usr/bin/env python3
"""Local entry to the Thor workflow: deploy, doctor, run, fetch."""
import argparse
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thor_kda.config import PACKAGE_ID, load_controller

PACKAGE_PATHS = (
    "tools", "skills", "infra", "knowledge", "thor_kda", "candidates", "kernels", "templates",
    "tasks", "patches", "docs", "sources.lock.json", "AGENTS.md", "README.md",
    "THIRD_PARTY_NOTICES.md", "requirements-dev.txt", "config/thor.example.json",
)
GENERATED_DIRECTORIES = {"results", "profile", "cache", "tmp", "build"}
EXCLUDED_DIRECTORIES = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"}


def remote(config, argv, **kwargs):
    return subprocess.run([*config.ssh_command(), shlex.join([str(x) for x in argv])], **kwargs)


def runtime_command(config, argv):
    return ["env", *(f"{key}={value}" for key, value in config.runtime.executor_environment().items()), *argv]


def package_bytes(runtime_only=False, extra_files=()):
    package = io.BytesIO()
    seen = set()
    with tarfile.open(fileobj=package, mode="w:gz") as archive:
        for name in (*PACKAGE_PATHS, *extra_files):
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Deployment files must be package-relative paths without traversal")
            if runtime_only and name in ("infra", "knowledge"):
                continue
            base = ROOT / name
            if not base.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError("Deployment file escapes package root")
            if not base.exists():
                continue
            paths = sorted(base.rglob("*")) if base.is_dir() else [base]
            for path in paths:
                relative = path.relative_to(ROOT)
                if any(part in EXCLUDED_DIRECTORIES for part in relative.parts):
                    continue
                if relative.parts[0] == "tasks" and any(part in GENERATED_DIRECTORIES for part in relative.parts[1:]):
                    continue
                if relative.parts[0] == "config" and str(relative) != "config/thor.example.json":
                    continue
                if path.is_file() and not path.is_symlink() and str(relative) not in seen:
                    if not path.resolve().is_relative_to(ROOT.resolve()):
                        raise ValueError("Deployment file resolves outside package root")
                    archive.add(path, arcname=str(relative), recursive=False)
                    seen.add(str(relative))
    return package.getvalue()


DEPLOY_CODE = r'''
import io, json, sys, tarfile
from pathlib import Path
root, allowed = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
package_id = sys.argv[3]
if allowed == Path('/') or root == allowed or not root.is_relative_to(allowed):
    raise SystemExit('Deployment must stay strictly below allowed_root')
marker = root / 'workflow.package.json'
if root.exists() and any(root.iterdir()):
    if not marker.is_file() or json.loads(marker.read_text()).get('package') != package_id:
        raise SystemExit('Destination exists without this package identity')
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()), mode='r:gz') as archive:
    members = archive.getmembers()
    for member in members:
        destination = (root / member.name).resolve()
        if not destination.is_relative_to(root) or destination == root or not member.isfile():
            raise SystemExit('Invalid deployment member: ' + member.name)
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({'package': package_id, 'stage': 'deploying'}))
    for member in members:
        destination = root / member.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.extractfile(member).read())
    marker.write_text(json.dumps({'package': package_id}))
print(json.dumps({'root': str(root), 'files': len(members)}))
'''


def deploy(config, runtime_only=False, extra_files=()):
    return remote(config, [config.runtime.python, "-B", "-c", DEPLOY_CODE,
                           config.remote_root, config.runtime.allowed_root, PACKAGE_ID],
                  input=package_bytes(runtime_only, extra_files)).returncode


FETCH_CODE = r'''
import json, sys
from pathlib import Path
root, allowed = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
if allowed == Path('/') or root == allowed or not root.is_relative_to(allowed):
    raise SystemExit('Fetch root escapes allowed_root')
marker = root / 'workflow.package.json'
if not marker.is_file() or json.loads(marker.read_text()).get('package') != sys.argv[4]:
    raise SystemExit('Missing package identity')
path = (root / sys.argv[3]).resolve()
if not path.is_relative_to(root) or not path.is_file():
    raise SystemExit('Requested file escapes package root or does not exist')
sys.stdout.buffer.write(path.read_bytes())
'''


def fetch(config, relative, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new local output path")
    result = remote(config, [config.runtime.python, "-B", "-c", FETCH_CODE,
                             config.remote_root, config.runtime.allowed_root, relative, PACKAGE_ID], capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(result.stdout)
    print(output)


def doctor_source():
    # Stream both modules so the read-only doctor also works before deployment.
    config_source = (ROOT / "thor_kda/config.py").read_text()
    prefix = (
        "import sys, types\n"
        "_package = types.ModuleType('thor_kda'); _package.__path__ = []\n"
        "sys.modules['thor_kda'] = _package\n"
        "_config = types.ModuleType('thor_kda.config')\n"
        "sys.modules['thor_kda.config'] = _config\n"
        f"exec({config_source!r}, _config.__dict__)\n"
    )
    return (prefix + (ROOT / "tools/remote_run.py").read_text()).encode()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Connection JSON (default: THOR_KDA_CONFIG or config/thor.local.json)")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    deploy_parser = subparsers.add_parser("deploy")
    deploy_parser.add_argument("--runtime-only", action="store_true")
    deploy_parser.add_argument("--extra-file", action="append", default=[], help="Package-relative reviewed file to include")
    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--output", type=Path)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--tag", required=True)
    run_parser.add_argument("--profile", choices=("none", "ncu", "nsys"), default="none")
    run_parser.add_argument("--ncu-kernel")
    run_parser.add_argument("--task-root", help="Absolute task directory on Thor (default: configured remote_root)")
    run_parser.add_argument("--cpu-only", action="store_true")
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    fetch_parser = subparsers.add_parser("fetch")
    fetch_parser.add_argument("remote_relative_path")
    fetch_parser.add_argument("output", type=Path)
    for child in (deploy_parser, doctor_parser, run_parser, fetch_parser):
        child.add_argument("--config", type=Path, default=argparse.SUPPRESS)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_controller(args.config)
        if args.mode == "deploy":
            return deploy(config, args.runtime_only, args.extra_file)
        if args.mode == "fetch":
            fetch(config, args.remote_relative_path, args.output)
            return 0
        if args.mode == "doctor":
            if args.output and args.output.exists():
                raise ValueError("Use a new local doctor output path")
            result = remote(config, runtime_command(config, [config.runtime.python, "-B", "-", "--doctor"]),
                            input=doctor_source(), capture_output=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("xb") as stream:
                    stream.write(result.stdout)
            print(result.stdout.decode())
            print(result.stderr.decode(), end="", file=sys.stderr)
            return result.returncode
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("run requires a command after --")
        argv = [config.runtime.python, "-B", str(config.remote_root / "tools/remote_run.py"),
                "--task-root", args.task_root or str(config.remote_root), "--tag", args.tag,
                "--profile", args.profile]
        if args.cpu_only:
            argv.append("--cpu-only")
        if args.ncu_kernel:
            argv.extend(["--ncu-kernel", args.ncu_kernel])
        return remote(config, runtime_command(config, [*argv, "--", *command])).returncode
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())

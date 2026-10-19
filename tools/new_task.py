#!/usr/bin/env python3
"""Prepare an independent Thor task contract without starting a GPU job."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]


def create(name, model, objective, workloads=None, *, arch="sm_110a", target_gpu="NVIDIA Thor", cpu="aarch64"):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name):
        raise ValueError("Use a lowercase task slug")
    task = ROOT / "tasks" / name
    if not task.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Task escapes workspace")
    if workloads is not None and not Path(workloads).is_file():
        raise ValueError("Workloads must reference an existing JSON file")
    task.mkdir(parents=True, exist_ok=False)
    for directory in ("baseline", "solution", "bench", "docs", "results", "profile", "candidates"):
        (task / directory).mkdir()
    contract = dict(
        schema_version=1, name=name, model=model, objective=objective,
        created_utc=datetime.now(timezone.utc).isoformat(),
        target=dict(gpu=target_gpu, arch=arch, cpu=cpu), remote_relative_root="tasks/" + name,
        baseline_manifest=None, workload_frozen=False, precision=None, graph=None,
        numerical_acceptance=None, model_quality_acceptance=None, performance_target=None,
        status="prepared; resolve acceptance from the requested scope and model before measurements",
    )
    (task / "task.json").write_text(json.dumps(contract, indent=2) + "\n")
    (task / "config.toml").write_text(
        f"[task]\nslug = {json.dumps(name)}\nmodel = {json.dumps(model)}\n"
        f"arch = {json.dumps(arch)}\ntarget_gpu = {json.dumps(target_gpu)}\n\n"
        "[benchmark]\nminimum_valid_pairs = 20\nmodel_e2e_required_for_promotion = true\n"
    )
    if workloads:
        shutil.copyfile(workloads, task / "bench/workloads.json")
    (task / "AGENTS.md").write_text(
        "Use ../../skills/thor-kda-workflow/SKILL.md. Freeze the accepted baseline, actual workload, "
        "and acceptance criteria before optimization. Append decisions and evidence to this task.\n"
    )
    (task / "prompt.md").write_text(
        f"# {name}\n\n{objective}\n\nTarget: {model} on {target_gpu}, {cpu}/{arch}.\n"
        "Read ../../skills/thor-kda-workflow/SKILL.md. Resolve the missing task.json fields "
        "from the requested scope and deployed model. Record parent candidate and source hashes "
        "before each change. Use model quality and unprofiled paired end-to-end measurements "
        "for adoption; retain profiler evidence separately.\n"
    )
    (task / "docs/decisions.jsonl").touch()
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--workloads", type=Path)
    parser.add_argument("--arch", default="sm_110a")
    args = parser.parse_args()
    print(create(args.name, args.model, args.objective, args.workloads, arch=args.arch))


if __name__ == "__main__":
    main()

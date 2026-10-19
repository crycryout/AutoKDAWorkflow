from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
from . import platform as plat
from .build import build_cutlass, dsl_environment
from .ledger import record, artifact
from .semantics import load

def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m thor_kda")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("probe"); p.add_argument("--output", type=Path, required=True)
    b = sp.add_parser("build-cutlass"); b.add_argument("--candidate", type=Path, required=True)
    v = sp.add_parser("validate"); v.add_argument("--candidate", type=Path, required=True)
    d = sp.add_parser("probe-cutedsl"); d.add_argument("--script", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "probe":
        plat.assert_root(a.output); a.output.parent.mkdir(parents=True, exist_ok=True)
        probe_record = plat.discover(); a.output.write_text(json.dumps(probe_record, indent=2, sort_keys=True) + "\n")
        print(json.dumps(probe_record, indent=2, sort_keys=True)); return 0
    if a.cmd == "build-cutlass":
        m = load(a.candidate); build_cutlass(a.candidate); print(json.dumps(m, indent=2)); return 0
    if a.cmd == "validate":
        m = load(a.candidate); env = dict(os.environ); env.update({"KDA_THOR_ROOT": str(plat.ROOT)})
        cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "tools/validate_cutlass.py"), "--candidate", str(a.candidate)]
        p = subprocess.run(cmd, env=env, text=True); return p.returncode
    if a.cmd == "probe-cutedsl":
        env = dsl_environment(); env["KDA_THOR_ROOT"] = str(plat.ROOT)
        p = subprocess.run([sys.executable, str(a.script)], env=env, text=True)
        record("cute_dsl_probe", {"returncode": p.returncode, "script": artifact(a.script)})
        return p.returncode
    return 2

if __name__ == "__main__": raise SystemExit(main())

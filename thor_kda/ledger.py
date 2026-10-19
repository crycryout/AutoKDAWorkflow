from __future__ import annotations
import json, time
from pathlib import Path
from .platform import ROOT, assert_root, sha256

def record(kind: str, payload: dict) -> Path:
    path = ROOT / "results/ledger.jsonl"; assert_root(path); path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind, **payload}
    with path.open("a") as f: f.write(json.dumps(row, sort_keys=True) + "\n")
    return path

def artifact(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}

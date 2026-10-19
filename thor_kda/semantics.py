from __future__ import annotations
import json
import re
from pathlib import Path

REQUIRED = ("name", "m", "n", "k", "dtype", "accum_dtype", "layout", "bias", "stream", "graph")

def load(path: Path) -> dict:
    data = json.loads(path.read_text())
    missing = [k for k in REQUIRED if k not in data]
    if missing: raise ValueError(f"candidate manifest missing: {missing}")
    if not isinstance(data["name"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", data["name"]):
        raise ValueError("candidate name must be a filename-safe identifier")
    if any(int(data[k]) <= 0 for k in ("m", "n", "k")): raise ValueError("M/N/K must be positive")
    if data["dtype"] != "bf16" or data["accum_dtype"] != "fp32":
        raise ValueError("This BF16 GEMM adapter requires BF16 input/output with FP32 accumulation")
    if data["layout"] not in ("row_row", "row_col"): raise ValueError("unsupported layout contract")
    if not isinstance(data["bias"], bool): raise ValueError("bias must be a boolean semantic contract")
    defines = data.get("defines", {})
    if int(defines.get("BIAS", int(data["bias"]))) != int(data["bias"]):
        raise ValueError("manifest bias and compiled BIAS define disagree")
    for shape in data.get("validation_shapes", []):
        if len(shape) != 3 or any(int(v) <= 0 for v in shape):
            raise ValueError(f"invalid validation shape: {shape}")
    return data

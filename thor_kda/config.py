"""Portable controller configuration and the explicit executor environment."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys


PACKAGE_ID = "auto-kda-workflow"


def _absolute(value, name):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path on Thor")
    if path == Path("/"):
        raise ValueError(f"{name} must not be the filesystem root")
    if ".." in path.parts:
        raise ValueError(f"{name} must not contain parent traversal")
    return path


@dataclass(frozen=True)
class RuntimeConfig:
    allowed_root: Path | None = None
    python: str = sys.executable
    cuda_home: Path = Path("/usr/local/cuda")
    expected_hostname: str | None = None
    expected_arch: str = "aarch64"
    expected_gpu: str = "NVIDIA Thor"
    cute_arch: str = "sm_110a"
    ncu: str | None = None
    nsys: str = "nsys"
    ncu_sections: str | None = None
    profiler_sudo: bool = True

    def __post_init__(self):
        for name in ("python", "expected_arch", "expected_gpu", "cute_arch", "nsys"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")

    @property
    def ncu_executable(self):
        return self.ncu or str(self.cuda_home / "bin/ncu")

    @classmethod
    def from_environ(cls, environ=None):
        env = os.environ if environ is None else environ
        raw_root = env.get("THOR_KDA_ALLOWED_ROOT")
        sudo = env.get("THOR_KDA_PROFILER_SUDO", "1")
        if sudo not in ("0", "1"):
            raise ValueError("THOR_KDA_PROFILER_SUDO must be 0 or 1")
        return cls(
            allowed_root=_absolute(raw_root, "THOR_KDA_ALLOWED_ROOT") if raw_root else None,
            python=env.get("THOR_KDA_PYTHON", sys.executable),
            cuda_home=_absolute(env.get("THOR_KDA_CUDA_HOME", "/usr/local/cuda"), "cuda_home"),
            expected_hostname=env.get("THOR_KDA_EXPECTED_HOSTNAME") or None,
            expected_arch=env.get("THOR_KDA_EXPECTED_ARCH", "aarch64"),
            expected_gpu=env.get("THOR_KDA_EXPECTED_GPU", "NVIDIA Thor"),
            cute_arch=env.get("THOR_KDA_CUTE_ARCH", "sm_110a"),
            ncu=env.get("THOR_KDA_NCU") or None,
            nsys=env.get("THOR_KDA_NSYS", "nsys"),
            ncu_sections=env.get("THOR_KDA_NCU_SECTIONS") or None,
            profiler_sudo=sudo == "1",
        )

    def executor_environment(self):
        if self.allowed_root is None:
            raise ValueError("Set allowed_root before running or deploying")
        return {
            "THOR_KDA_ALLOWED_ROOT": str(self.allowed_root),
            "THOR_KDA_PYTHON": self.python,
            "THOR_KDA_CUDA_HOME": str(self.cuda_home),
            "THOR_KDA_EXPECTED_HOSTNAME": self.expected_hostname or "",
            "THOR_KDA_EXPECTED_ARCH": self.expected_arch,
            "THOR_KDA_EXPECTED_GPU": self.expected_gpu,
            "THOR_KDA_CUTE_ARCH": self.cute_arch,
            "THOR_KDA_NCU": self.ncu or "",
            "THOR_KDA_NSYS": self.nsys,
            "THOR_KDA_NCU_SECTIONS": self.ncu_sections or "",
            "THOR_KDA_PROFILER_SUDO": "1" if self.profiler_sudo else "0",
        }


@dataclass(frozen=True)
class ControllerConfig:
    ssh_target: str
    remote_root: Path
    runtime: RuntimeConfig
    ssh_prefix: tuple = ()
    ssh_port: int = 22
    ssh_identity_file: str | None = None
    ssh_known_hosts_file: str | None = None

    def ssh_command(self):
        argv = [*self.ssh_prefix, "ssh", "-T", "-o", "BatchMode=yes", "-o",
                "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15", "-o",
                "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "-p", str(self.ssh_port)]
        if self.ssh_identity_file:
            argv.extend(["-i", str(Path(self.ssh_identity_file).expanduser())])
        if self.ssh_known_hosts_file:
            argv.extend(["-o", "UserKnownHostsFile=" + str(Path(self.ssh_known_hosts_file).expanduser())])
        return [*argv, self.ssh_target]


def load_controller(path=None, *, root=None, environ=None):
    env = os.environ if environ is None else environ
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    source = Path(path or env.get("THOR_KDA_CONFIG") or base / "config/thor.local.json").expanduser()
    if not source.is_file():
        raise ValueError(f"Configuration not found: {source}. Copy config/thor.example.json and set your Thor connection.")
    data = json.loads(source.read_text())
    runtime_keys = {"python", "cuda_home", "expected_hostname", "expected_arch", "expected_gpu",
                    "cute_arch", "ncu", "nsys", "ncu_sections", "profiler_sudo"}
    known = runtime_keys | {"allowed_root", "remote_root", "ssh_target", "ssh_prefix", "ssh_port",
                            "ssh_identity_file", "ssh_known_hosts_file"}
    if not isinstance(data, dict) or set(data) - known:
        raise ValueError("Configuration contains unsupported fields")
    for name in ("ssh_target", "remote_root", "allowed_root"):
        if not isinstance(data.get(name), str) or not data[name].strip():
            raise ValueError(f"Configuration requires {name}")
    if data["ssh_target"].startswith("-") or any(c.isspace() for c in data["ssh_target"]):
        raise ValueError("ssh_target must be one SSH alias or user@host")
    allowed = _absolute(data["allowed_root"], "allowed_root")
    remote_root = _absolute(data["remote_root"], "remote_root")
    if not remote_root.is_relative_to(allowed) or remote_root == allowed:
        raise ValueError("remote_root must be a task directory strictly below allowed_root")
    prefix = data.get("ssh_prefix", [])
    if not isinstance(prefix, list) or any(not isinstance(x, str) or not x for x in prefix):
        raise ValueError("ssh_prefix must be a list of command arguments")
    port = data.get("ssh_port", 22)
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ValueError("ssh_port must be between 1 and 65535")
    if "profiler_sudo" in data and not isinstance(data["profiler_sudo"], bool):
        raise ValueError("profiler_sudo must be a boolean")
    for key in runtime_keys - {"profiler_sudo"}:
        value = data.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key} must be a nonempty string or null")
    runtime_data = {k: v for k, v in data.items() if k in runtime_keys and v is not None}
    runtime_data.setdefault("python", "python3")
    runtime_data["cuda_home"] = _absolute(data.get("cuda_home", "/usr/local/cuda"), "cuda_home")
    runtime = RuntimeConfig(allowed_root=allowed, **runtime_data)
    return ControllerConfig(data["ssh_target"], remote_root, runtime, tuple(prefix), port,
                            data.get("ssh_identity_file"), data.get("ssh_known_hosts_file"))

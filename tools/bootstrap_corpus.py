#!/usr/bin/env python3
"""Install pinned public knowledge corpora and reviewed Thor adaptations.

Runs on analysis hosts as well as Thor. No GPU or elevated privileges are needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
from urllib.parse import urlparse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
STAMP = ".auto-kda-source.json"
DEFAULT_SOURCES = ("ai_infra", "kernel_wiki", "ncu_report_skill")
OPTIONAL_SOURCES = ("kda", "kda_pilot")
MAX_EXPANDED_BYTES = 2 * 1024**3
MAX_ARCHIVE_MEMBERS = 250_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Unsafe relative path: {value!r}")
    if not path.parts or str(path) != value or any(p in {"", "."} for p in path.parts):
        raise ValueError(f"Non-canonical relative path: {value!r}")
    return path


def ensure_no_symlink_parents(root: Path, path: Path) -> None:
    current = root
    for component in path.relative_to(root).parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"Refuse symbolic-link destination: {current}")
        if current.exists() and not current.is_dir():
            raise ValueError(f"Destination parent is not a directory: {current}")


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract one GitHub archive root, accepting regular files/directories only."""
    destination.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    archive_root = None
    total_bytes = 0
    count = 0
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            count += 1
            if count > MAX_ARCHIVE_MEMBERS:
                raise ValueError("Archive contains too many members")
            name = member.name.rstrip("/") if member.isdir() else member.name
            path = relative_path(name)
            if archive_root is None:
                archive_root = path.parts[0]
            if path.parts[0] != archive_root:
                raise ValueError("Archive contains more than one top-level directory")
            if not member.isdir() and not member.isfile():
                raise ValueError(f"Archive links and special files are not supported: {name}")
            if len(path.parts) == 1:
                if not member.isdir():
                    raise ValueError("Archive root must be a directory")
                continue
            relative = PurePosixPath(*path.parts[1:])
            if str(relative) in seen:
                raise ValueError(f"Duplicate archive member: {relative}")
            seen.add(str(relative))
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            total_bytes += member.size
            if member.size < 0 or total_bytes > MAX_EXPANDED_BYTES:
                raise ValueError("Archive exceeds the expanded-size limit")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f"Unreadable archive member: {name}")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)
    if not any(destination.iterdir()):
        raise ValueError("Archive is empty")


def tree_hashes(destination: Path) -> dict:
    files = {}
    for path in sorted(destination.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Unexpected symbolic link in installed source: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Unexpected special file in installed source: {path}")
        relative_path = path.relative_to(destination)
        relative = relative_path.as_posix()
        if "__pycache__" in relative_path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if relative == STAMP:
            continue
        files[relative] = {"sha256": sha256_file(path), "executable": bool(path.stat().st_mode & 0o111)}
    return files


def load_configuration(root: Path, with_kda: bool) -> tuple[dict, dict]:
    lock = json.loads((root / "sources.lock.json").read_text())
    if lock.get("schema_version") != 1 or not isinstance(lock.get("sources"), dict):
        raise ValueError("Unsupported sources.lock.json schema")
    manifest = json.loads((root / "patches/manifest.json").read_text())
    patches = {}
    for entry in manifest:
        name = entry["patch"]
        if len(relative_path(name).parts) != 1 or name in patches:
            raise ValueError("Invalid or duplicate patch name")
        path = root / "patches" / name
        if path.is_symlink() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Patch hash mismatch: {name}")
        patches[name] = entry
    keys = DEFAULT_SOURCES + (OPTIONAL_SOURCES if with_kda else ())
    sources = {key: lock["sources"][key] for key in keys}
    destinations = set()
    for key, pin in sources.items():
        if not re.fullmatch(r"[0-9a-f]{40}", pin["commit"]):
            raise ValueError(f"Source requires a full public commit: {key}")
        if not re.fullmatch(r"[0-9a-f]{64}", pin["archive_sha256"]):
            raise ValueError(f"Source requires an archive SHA-256: {key}")
        url = urlparse(pin["url"])
        if url.scheme != "https" or url.netloc != "github.com" or not re.fullmatch(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", url.path) or url.query or url.fragment:
            raise ValueError(f"Expected a public GitHub repository URL: {key}")
        relative_path(pin["destination"])
        if pin["destination"] in destinations:
            raise ValueError("Duplicate source destination")
        destinations.add(pin["destination"])
        if pin.get("patch"):
            patch = patches[pin["patch"]]
            for filename in patch["files"]:
                relative_path(filename)
                if not filename.startswith(pin["destination"] + "/"):
                    raise ValueError(f"Patch escapes its source destination: {filename}")
            patch_text = (root / "patches" / pin["patch"]).read_text()
            headers = re.findall(r"^(?:--- a/|\+\+\+ b/)(.+)$", patch_text, re.MULTILINE)
            if not headers or set(headers) != set(patch["files"]):
                raise ValueError("Patch file list differs from its manifest")
    return sources, patches


def cached_archive(root: Path, key: str, pin: dict) -> Path:
    cache = root / "cache/upstream"
    ensure_no_symlink_parents(root, cache)
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"{key}-{pin['commit']}.tar.gz"
    if archive.is_symlink():
        raise ValueError(f"Refuse symbolic-link archive: {archive}")
    if archive.exists():
        if sha256_file(archive) != pin["archive_sha256"]:
            raise ValueError(f"Cached archive hash mismatch: {key}")
        return archive
    url = pin["url"].replace("https://github.com/", "https://codeload.github.com/") + "/tar.gz/" + pin["commit"]
    descriptor, temporary = tempfile.mkstemp(prefix=f".{key}-", suffix=".partial", dir=cache)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(url, timeout=60) as response:
            shutil.copyfileobj(response, output)
        if sha256_file(temporary) != pin["archive_sha256"]:
            raise ValueError(f"Downloaded archive hash mismatch: {key}")
        # A hard link atomically publishes a regular file without replacing any path.
        os.link(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def expected_identity(key: str, pin: dict, patches: dict) -> dict:
    return {
        "schema_version": 1,
        "source": key,
        "url": pin["url"],
        "commit": pin["commit"],
        "destination": pin["destination"],
        "archive_sha256": pin["archive_sha256"],
        "patch_sha256": patches[pin["patch"]]["sha256"] if pin.get("patch") else None,
    }


def verify_installed(root: Path, key: str, pin: dict, patches: dict) -> dict:
    destination = root / pin["destination"]
    ensure_no_symlink_parents(root, destination)
    stamp = destination / STAMP
    if stamp.is_symlink() or not stamp.is_file():
        raise ValueError(f"Refuse existing source without a valid stamp: {destination}")
    record = json.loads(stamp.read_text())
    identity = expected_identity(key, pin, patches)
    if any(record.get(field) != value for field, value in identity.items()):
        raise ValueError(f"Installed source does not match locked identity: {key}")
    if not record.get("files") or record["files"] != tree_hashes(destination):
        raise ValueError(f"Installed source file hashes have drifted: {key}")
    return {"source": key, "destination": pin["destination"], "state": "verified", "files": len(record["files"])}


def install_source(root: Path, key: str, pin: dict, patches: dict, verify_only: bool = False) -> dict:
    destination = root / pin["destination"]
    ensure_no_symlink_parents(root, destination)
    if destination.exists():
        return verify_installed(root, key, pin, patches)
    if verify_only:
        raise ValueError(f"Source is not installed: {pin['destination']}")
    archive = cached_archive(root, key, pin)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=".bootstrap-stage-", dir=destination.parent))
    try:
        staged_source = stage_root / pin["destination"]
        staged_source.parent.mkdir(parents=True, exist_ok=True)
        safe_extract(archive, staged_source)
        if pin.get("patch"):
            patch = (root / "patches" / pin["patch"]).resolve()
            for check in (True, False):
                command = ["git", "apply", "--no-index"]
                if check:
                    command.append("--check")
                subprocess.run(command + [str(patch)], cwd=stage_root, check=True, capture_output=True, text=True)
        record = expected_identity(key, pin, patches)
        record["files"] = tree_hashes(staged_source)
        (staged_source / STAMP).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        ensure_no_symlink_parents(root, destination)
        if destination.exists():
            raise ValueError(f"Destination appeared during installation: {destination}")
        staged_source.rename(destination)
        return {"source": key, "destination": pin["destination"], "state": "installed", "files": len(record["files"])}
    finally:
        shutil.rmtree(stage_root)


def bootstrap(root: Path = ROOT, *, with_kda: bool = False, verify_only: bool = False) -> list[dict]:
    root = root.resolve(strict=True)
    sources, patches = load_configuration(root, with_kda)
    lock = root / ".bootstrap.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError("Another bootstrap is active; inspect .bootstrap.lock before retrying") from error
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(str(os.getpid()) + "\n")
        return [install_source(root, key, pin, patches, verify_only) for key, pin in sources.items()]
    finally:
        lock.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-kda", action="store_true", help="Also install public KDA and KDA-Pilot under upstream/")
    parser.add_argument("--verify-only", action="store_true", help="Verify installed identities and every file hash without downloading")
    parser.add_argument("--root", type=Path, default=ROOT, help="Workflow checkout containing sources.lock.json")
    args = parser.parse_args()
    try:
        records = bootstrap(args.root, with_kda=args.with_kda, verify_only=args.verify_only)
    except (ValueError, OSError, subprocess.CalledProcessError, tarfile.TarError) as error:
        parser.exit(1, f"bootstrap: {error}\n")
    print(json.dumps({"sources": records}, indent=2))


if __name__ == "__main__":
    main()

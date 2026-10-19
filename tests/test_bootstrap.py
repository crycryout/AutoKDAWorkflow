"""CPU-only installation and filesystem safety checks using tiny local archives."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


corpus = load_tool("bootstrap_corpus")
links = load_tool("install_links")


def archive(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as handle:
        for name, payload, kind in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.mode = 0o644
            if kind == tarfile.REGTYPE:
                member.size = len(payload)
                handle.addfile(member, io.BytesIO(payload))
            else:
                if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                    member.linkname = "../../outside"
                handle.addfile(member)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "patches").mkdir()
    sources = {}
    for key, destination in zip(corpus.DEFAULT_SOURCES, ("infra", "knowledge/KernelWiki", "knowledge/ncu-report-skill")):
        pin = {"url": f"https://github.com/example/{key}", "commit": "a" * 40, "destination": destination}
        path = root / "cache/upstream" / f"{key}-{pin['commit']}.tar.gz"
        pin["archive_sha256"] = archive(path, [("source/README.md", b"baseline\n", tarfile.REGTYPE)])
        sources[key] = pin
    patch_name = "thor.patch"
    text = "--- a/infra/README.md\n+++ b/infra/README.md\n@@ -1 +1 @@\n-baseline\n+Thor\n"
    (root / "patches" / patch_name).write_text(text)
    sources["ai_infra"]["patch"] = patch_name
    (root / "patches/manifest.json").write_text(json.dumps([{"patch": patch_name, "sha256": hashlib.sha256(text.encode()).hexdigest(), "files": ["infra/README.md"]}]))
    (root / "sources.lock.json").write_text(json.dumps({"schema_version": 1, "sources": sources}))
    return root


def test_install_patch_idempotence_and_offline_verification(checkout):
    with patch.object(corpus.urllib.request, "urlopen", side_effect=AssertionError("network is unnecessary")):
        first = corpus.bootstrap(checkout)
        assert [row["state"] for row in first] == ["installed"] * 3
        assert (checkout / "infra/README.md").read_text() == "Thor\n"
        assert all(row["state"] == "verified" for row in corpus.bootstrap(checkout))
        assert all(row["state"] == "verified" for row in corpus.bootstrap(checkout, verify_only=True))
    assert not (checkout / ".bootstrap.lock").exists()


@pytest.mark.parametrize("change", ["modify", "add", "remove", "link", "chmod"])
def test_verify_rejects_source_drift(checkout, change):
    corpus.bootstrap(checkout)
    readme = checkout / "infra/README.md"
    if change == "modify":
        readme.write_text("different\n")
    elif change == "add":
        (readme.parent / "extra.txt").write_text("extra")
    elif change == "remove":
        readme.unlink()
    elif change == "link":
        readme.unlink()
        readme.symlink_to("/etc/passwd")
    else:
        readme.chmod(0o755)
    with pytest.raises(ValueError, match="drifted|symbolic link"):
        corpus.bootstrap(checkout, verify_only=True)


def test_existing_directory_is_never_replaced(checkout):
    destination = checkout / "infra"
    destination.mkdir()
    (destination / "owned.txt").write_text("keep")
    with pytest.raises(ValueError, match="without a valid stamp"):
        corpus.bootstrap(checkout)
    assert (destination / "owned.txt").read_text() == "keep"


def test_verify_only_does_not_install(checkout):
    with patch.object(corpus.urllib.request, "urlopen", side_effect=AssertionError("no download")):
        with pytest.raises(ValueError, match="not installed"):
            corpus.bootstrap(checkout, verify_only=True)
    assert not (checkout / "infra").exists()


def test_patch_hash_mismatch_prevents_install(checkout):
    with (checkout / "patches/thor.patch").open("a") as handle:
        handle.write("corruption\n")
    with pytest.raises(ValueError, match="Patch hash mismatch"):
        corpus.bootstrap(checkout)
    assert not (checkout / "infra").exists()


def test_archive_hash_mismatch_prevents_install(checkout):
    path = next((checkout / "cache/upstream").glob("ai_infra*"))
    path.write_bytes(b"corruption")
    with pytest.raises(ValueError, match="archive hash mismatch"):
        corpus.bootstrap(checkout)
    assert not (checkout / "infra").exists()


def test_patch_failure_cleans_stage_and_leaves_destination_absent(checkout):
    path = checkout / "patches/thor.patch"
    path.write_text(path.read_text().replace("-baseline", "-missing"))
    manifest = json.loads((checkout / "patches/manifest.json").read_text())
    manifest[0]["sha256"] = corpus.sha256_file(path)
    (checkout / "patches/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(corpus.subprocess.CalledProcessError):
        corpus.bootstrap(checkout)
    assert not (checkout / "infra").exists()
    assert not list(checkout.rglob(".bootstrap-stage-*"))
    assert not (checkout / ".bootstrap.lock").exists()


def test_destination_parent_symlink_is_rejected(checkout, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (checkout / "knowledge").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic-link destination"):
        corpus.bootstrap(checkout)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("entries", [
    [("source/../escape", b"x", tarfile.REGTYPE)],
    [("/absolute", b"x", tarfile.REGTYPE)],
    [("source/link", b"", tarfile.SYMTYPE)],
    [("source/link", b"", tarfile.LNKTYPE)],
    [("source/device", b"", tarfile.CHRTYPE)],
    [("source/a", b"a", tarfile.REGTYPE), ("other/b", b"b", tarfile.REGTYPE)],
    [("source/a", b"a", tarfile.REGTYPE), ("source/a", b"b", tarfile.REGTYPE)],
    [("source\\escape", b"x", tarfile.REGTYPE)],
])
def test_unsafe_archives_are_rejected(tmp_path, entries):
    source = tmp_path / "input.tar.gz"
    archive(source, entries)
    with pytest.raises(ValueError):
        corpus.safe_extract(source, tmp_path / "output")
    assert not (tmp_path / "escape").exists()


def test_download_failure_cleans_partial(checkout):
    pins, _ = corpus.load_configuration(checkout, False)
    pin = pins["ai_infra"]
    path = checkout / "cache/upstream" / f"ai_infra-{pin['commit']}.tar.gz"
    path.unlink()
    with patch.object(corpus.urllib.request, "urlopen", side_effect=OSError("download failed")):
        with pytest.raises(OSError, match="download failed"):
            corpus.bootstrap(checkout)
    assert not list((checkout / "cache/upstream").glob("*.partial"))


@pytest.fixture
def skill_checkout(tmp_path):
    root = tmp_path / "workflow"
    skill = root / "skills/thor-kda-workflow"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: thor-kda-workflow\ndescription: Thor workflow\n---\n")
    return root


def test_links_use_home_preview_and_idempotent_apply(skill_checkout, tmp_path, monkeypatch):
    home = tmp_path / "agent-home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    preview = links.install(root=skill_checkout)
    assert not home.exists()
    assert preview["skills"] == 1
    assert len(preview["entries"]) == 2
    assert all(str(home) in row["path"] for row in preview["entries"])
    applied = links.install(root=skill_checkout, apply=True)
    assert all(Path(row["path"]).is_symlink() for row in applied["entries"])
    assert all(row["state"] == "already linked" for row in links.install(root=skill_checkout, apply=True)["entries"])


def test_links_preflight_all_collisions(skill_checkout, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    second.mkdir()
    collision = second / "thor-kda-workflow"
    collision.write_text("keep")
    with pytest.raises(ValueError, match="collision"):
        links.install([first, second], root=skill_checkout, apply=True)
    assert not first.exists()
    assert collision.read_text() == "keep"


def test_links_reject_duplicate_names(skill_checkout, tmp_path):
    duplicate = skill_checkout / "knowledge/extra"
    duplicate.mkdir(parents=True)
    (duplicate / "SKILL.md").write_text("---\nname: thor-kda-workflow\n---\n")
    with pytest.raises(ValueError, match="Duplicate skill name"):
        links.install([tmp_path / "agent"], root=skill_checkout)


def test_links_reject_duplicate_destinations(skill_checkout, tmp_path):
    destination = tmp_path / "agent"
    with pytest.raises(ValueError, match="Duplicate destination"):
        links.install([destination, destination], root=skill_checkout)


def test_links_reject_unsafe_name(skill_checkout, tmp_path):
    (skill_checkout / "skills/thor-kda-workflow/SKILL.md").write_text("---\nname: ../../escape\n---\n")
    with pytest.raises(ValueError, match="Invalid skill name"):
        links.install([tmp_path / "agent"], root=skill_checkout)


def test_verify_ignores_python_bytecode_cache(checkout):
    corpus.bootstrap(checkout)
    cached = checkout / "infra/__pycache__/helpers.cpython-312.pyc"
    cached.parent.mkdir()
    cached.write_bytes(b"runtime cache")
    assert all(row["state"] == "verified" for row in corpus.bootstrap(checkout, verify_only=True))

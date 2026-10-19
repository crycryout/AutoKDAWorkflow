#!/usr/bin/env python3
"""Preview or install agent skill links for this checkout without replacing files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]


def skills(root: Path = ROOT) -> list[tuple[str, Path]]:
    roots = (root / "skills", root / "infra/skills", root / "infra/model-pr-optimization-history", root / "knowledge")
    found = []
    seen = set()
    for base in roots:
        if not base.exists():
            continue
        for path in sorted(base.rglob("SKILL.md")):
            if path.is_symlink():
                raise ValueError(f"Skill metadata must be a regular file: {path}")
            content = path.read_text(encoding="utf-8")
            parts = content.split("---", 2)
            if len(parts) != 3 or parts[0].strip():
                raise ValueError(f"Skill requires YAML frontmatter: {path}")
            frontmatter = yaml.safe_load(parts[1])
            name = frontmatter.get("name") if isinstance(frontmatter, dict) else None
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in {".", ".."}:
                raise ValueError(f"Invalid skill name in {path}")
            if name in seen:
                raise ValueError(f"Duplicate skill name: {name}")
            seen.add(name)
            target = path.parent.resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError(f"Skill target escapes this checkout: {target}")
            found.append((name, target))
    if not found:
        raise ValueError("No local skills found; ensure the workflow skill is present")
    return found


def install(destinations: list[Path] | None = None, *, apply: bool = False, root: Path = ROOT) -> dict:
    root = root.resolve()
    entries = skills(root)
    destinations = destinations or [Path.home() / ".codex/skills", Path.home() / ".claude/skills"]
    resolved = [path.expanduser().absolute() for path in destinations]
    if len({str(path.resolve()) for path in resolved}) != len(resolved):
        raise ValueError("Duplicate destination directories")
    plan = []
    for base in resolved:
        if base.exists() and not base.is_dir():
            raise ValueError(f"Skill destination is not a directory: {base}")
        for name, target in entries:
            link = base / name
            if link.exists() or link.is_symlink():
                if not link.is_symlink() or link.resolve() != target:
                    raise ValueError(f"Existing skill collision: {link}")
                state = "already linked"
            else:
                state = "create link"
            plan.append({"path": str(link), "target": str(target), "state": state})
    created = []
    if apply:
        try:
            for row in plan:
                if row["state"] == "create link":
                    link = Path(row["path"])
                    link.parent.mkdir(parents=True, exist_ok=True)
                    link.symlink_to(row["target"], target_is_directory=True)
                    created.append(link)
        except OSError:
            # Roll back this invocation's links only; never remove a pre-existing link.
            for link in reversed(created):
                if link.is_symlink():
                    link.unlink()
            raise
    return {"applied": apply, "skills": len(entries), "entries": plan}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Create links after validating every destination; default is preview")
    parser.add_argument("--destination", type=Path, action="append", help="Agent skills directory; repeat to select multiple agents")
    args = parser.parse_args()
    try:
        result = install(args.destination, apply=args.apply)
    except (ValueError, OSError, yaml.YAMLError) as error:
        parser.exit(1, f"install-links: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

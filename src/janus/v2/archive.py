"""Build a complete, deterministic post-evaluation reproduction directory."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _committed_at(project_root: Path, commit: str, path: Path) -> bool:
    relative = path.resolve().relative_to(project_root.resolve())
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:{relative}"],
        cwd=project_root,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _copy(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _secret_scan(root: Path) -> list[str]:
    patterns = (
        re.compile(rb"sk-ant-[A-Za-z0-9_-]{16,}"),
        re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
        re.compile(rb"AIza[A-Za-z0-9_-]{20,}"),
    )
    findings = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix in {".pkl", ".png", ".pdf"}:
            continue
        data = path.read_bytes()
        if any(pattern.search(data) for pattern in patterns):
            findings.append(str(path.relative_to(root)))
    return findings


def build_reproduction_archive(
    project_root: Path,
    output_directory: Path,
    response_manifests: tuple[Path, ...],
    *,
    response_commit: str,
) -> dict[str, Any]:
    if len(response_manifests) != 4:
        raise ValueError("archive requires exactly four sealed response manifests")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError("archive output must not already contain files")
    manifests = [_read(path) for path in response_manifests]
    if not all(_committed_at(project_root, response_commit, path) for path in response_manifests):
        raise ValueError("all response manifests must exist in the declared pre-unblinding commit")
    if any(manifest.get("unblinded", True) for manifest in manifests):
        raise ValueError("response manifests must state that collection remained blinded")

    required_sources = {
        "LICENSE": project_root / "LICENSE",
        "CITATION.cff": project_root / "CITATION.cff",
        "pyproject.toml": project_root / "pyproject.toml",
        "uv.lock": project_root / "uv.lock",
        "Dockerfile": project_root / "Dockerfile",
        "src": project_root / "src",
        "tests": project_root / "tests",
        "configs": project_root / "configs",
        "docs": project_root / "docs",
        "manifests": project_root / "manifests",
        "private/hidden-v2": project_root / "artifacts/hidden-v2",
        "analysis/comparative-report.json": (
            project_root / "artifacts/hidden-v2-analysis/comparative-report.json"
        ),
        "reports": project_root / "reports/v2-publication",
        ".github": project_root / ".github",
    }
    missing = [name for name, source in required_sources.items() if not source.exists()]
    if missing:
        raise ValueError(f"archive sources are incomplete: {missing}")
    output_directory.mkdir(parents=True, exist_ok=True)
    for destination, source in required_sources.items():
        _copy(source, output_directory / destination)
    for index, (path, manifest) in enumerate(zip(response_manifests, manifests, strict=True)):
        role = str(manifest.get("model_role", f"model-{index + 1}"))
        source_directory = path.parent.parent if path.parent.name == "manifests" else path.parent
        _copy(source_directory, output_directory / "responses" / role)

    secret_findings = _secret_scan(output_directory)
    if secret_findings:
        raise ValueError(f"possible API credentials found in archive: {secret_findings}")
    hashes = {
        str(path.relative_to(output_directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_directory.rglob("*"))
        if path.is_file() and path.name != "archive-manifest.json"
    }
    manifest = {
        "schema_version": "janus-v2-reproduction-archive-v1",
        "response_commit": response_commit,
        "response_manifest_count": 4,
        "contains_revealed_private_data": True,
        "secret_scan_findings": [],
        "sha256": hashes,
    }
    (output_directory / "archive-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    return manifest

"""Canonical cache identity and immutable file-backed trial cache."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from janus.models.base import ModelRequest


def request_cache_key(request: ModelRequest) -> str:
    digest = hashlib.sha256()
    fields = (
        request.settings.model_dump_json().encode(),
        request.context.model_dump_json().encode(),
        request.system_input,
        request.developer_input,
        request.user_input,
        request.output_schema,
        request.logical_trial_id.encode(),
        b"repair" if request.repair_attempt else b"primary",
    )
    for value in fields:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


class FileTrialCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict[str, object] | None:
        path = self.directory / f"{key}.json"
        return None if not path.exists() else json.loads(path.read_text())

    def put(self, key: str, payload: dict[str, object]) -> Path:
        path = self.directory / f"{key}.json"
        if path.exists():
            existing = json.loads(path.read_text())
            if existing != payload:
                raise ValueError("refusing to overwrite an existing logical trial")
            return path
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
        return path

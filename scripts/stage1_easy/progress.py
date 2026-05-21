#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def default_progress_path() -> Path:
    return Path("outputs/stage1_easy/monitor/stage1_easy.progress.json")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time()))


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_progress(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def update_progress(path: Path, step: str, patch: dict[str, Any]) -> dict[str, Any]:
    st = read_progress(path)
    st.setdefault("schema_version", 1)
    st["updated_at"] = _now_iso()
    st.setdefault("steps", {})
    st["steps"].setdefault(step, {})
    st["steps"][step].update(patch)
    _atomic_write_json(path, st)
    return st


def mark_running(path: Path, step: str, **extra: Any) -> None:
    update_progress(
        path,
        step,
        {
            "status": "running",
            "started_at": _now_iso(),
            "ended_at": None,
            "error": None,
            **extra,
        },
    )


def mark_done(path: Path, step: str, **extra: Any) -> None:
    update_progress(path, step, {"status": "done", "ended_at": _now_iso(), **extra})


def mark_error(path: Path, step: str, error: str, **extra: Any) -> None:
    update_progress(path, step, {"status": "error", "ended_at": _now_iso(), "error": error, **extra})


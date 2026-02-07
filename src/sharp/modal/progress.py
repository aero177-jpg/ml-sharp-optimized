"""Progress storage utilities for Modal polling."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

PROGRESS_VOLUME_NAME = "sharp-progress"
PROGRESS_VOLUME_PATH = "/cache/progress"

progress_volume = modal.Volume.from_name(PROGRESS_VOLUME_NAME, create_if_missing=True)

_SAFE_JOB_ID = re.compile(r"[^a-zA-Z0-9._-]+")


def _sanitize_job_id(job_id: str) -> str:
    if not job_id:
        raise ValueError("job_id is required")
    return _SAFE_JOB_ID.sub("_", job_id)


def _job_path(job_id: str) -> Path:
    safe_id = _sanitize_job_id(job_id)
    return Path(PROGRESS_VOLUME_PATH) / f"{safe_id}.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_progress(job_id: str, data: dict[str, Any], *, commit: bool = True) -> dict[str, Any]:
    """Write a progress snapshot for a job.

    Args:
        job_id: Job identifier.
        data: Progress fields to store.
        commit: Whether to commit the Modal volume.

    Returns:
        The stored progress snapshot.
    """
    path = _job_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = dict(data)
    snapshot["job_id"] = job_id
    snapshot["updated_at"] = _now_iso()

    step = snapshot.get("step")
    total_steps = snapshot.get("total_steps")
    if isinstance(step, (int, float)) and isinstance(total_steps, (int, float)) and total_steps:
        snapshot["percent"] = int(max(0.0, min(100.0, 100.0 * float(step) / float(total_steps))))

    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(snapshot), encoding="utf-8")
    temp_path.replace(path)

    if commit:
        progress_volume.commit()

    return snapshot


def read_progress(job_id: str) -> dict[str, Any] | None:
    """Read the progress snapshot for a job.

    Reloads the volume first so we see the latest writes from other containers.
    """
    progress_volume.reload()

    path = _job_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

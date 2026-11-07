"""Job progress and cancellation state for Modal polling.

Progress snapshots and cancel flags live in a ``modal.Dict``: every write is
immediately visible to every container, with no Volume commit/reload round
trip. The frontend polls this on every tick, and the GPU loop checks the
cancel flag between images, so both paths stay cheap.

Result *files* still live on ``progress_volume`` (see ``sharp.modal.api``).

The public functions keep their old signatures. ``commit=`` arguments are
accepted and ignored, since Dict writes need no commit.
"""

from __future__ import annotations

import re
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import modal

# Volume that holds temporary result files (not progress any more).
PROGRESS_VOLUME_NAME = "sharp-progress"
PROGRESS_VOLUME_PATH = "/cache/progress"
progress_volume = modal.Volume.from_name(PROGRESS_VOLUME_NAME, create_if_missing=True)

# Progress + cancel flags. Entries expire on their own after 7 days without access.
PROGRESS_DICT_NAME = "sharp-job-progress"
progress_store = modal.Dict.from_name(PROGRESS_DICT_NAME, create_if_missing=True)

_SAFE_JOB_ID = re.compile(r"[^a-zA-Z0-9._-]+")
# Serializes read-merge-write per job inside one container.
_WRITE_LOCKS: dict[str, threading.Lock] = defaultdict(threading.Lock)

_TERMINAL_STATUSES = {"cancelled"}


def _sanitize_job_id(job_id: str) -> str:
    if not job_id:
        raise ValueError("job_id is required")
    return _SAFE_JOB_ID.sub("_", job_id)


def _job_key(job_id: str) -> str:
    return f"job:{_sanitize_job_id(job_id)}"


def _cancel_key(job_id: str) -> str:
    return f"cancel:{_sanitize_job_id(job_id)}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_progress(job_id: str, data: dict[str, Any], *, commit: bool = True) -> dict[str, Any]:
    """Merge ``data`` into the job's progress snapshot and store it.

    A job marked ``cancelled`` stays cancelled: late writes from the worker
    (an upload finishing after the user cancelled) cannot flip it back.
    """
    del commit  # Dict writes are visible immediately.
    key = _job_key(job_id)
    with _WRITE_LOCKS[job_id]:
        existing = progress_store.get(key)
        snapshot: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
        if (
            snapshot.get("status") in _TERMINAL_STATUSES
            and data.get("status") not in _TERMINAL_STATUSES
        ):
            # Keep the terminal status but let file lists etc. update.
            data = {k: v for k, v in data.items() if k not in {"status", "phase", "done", "message"}}
        snapshot.update(data)
        snapshot["job_id"] = job_id
        snapshot["updated_at"] = _now_iso()

        step = snapshot.get("step")
        total_steps = snapshot.get("total_steps")
        if isinstance(step, (int, float)) and isinstance(total_steps, (int, float)) and total_steps:
            snapshot["percent"] = int(max(0.0, min(100.0, 100.0 * float(step) / float(total_steps))))

        progress_store.put(key, snapshot)
    return snapshot


def write_status(
    job_id: str,
    *,
    status: str,
    phase: str | None = None,
    message: str | None = None,
    step: int | float | None = None,
    total_steps: int | float | None = None,
    commit: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    """Write a normalized status payload for frontend polling."""
    payload: dict[str, Any] = {"status": status}
    if phase is not None:
        payload["phase"] = phase
    if message is not None:
        payload["message"] = message
    if step is not None:
        payload["step"] = step
    if total_steps is not None:
        payload["total_steps"] = total_steps
    payload.update(extra)
    return write_progress(job_id, payload, commit=commit)


def read_progress(job_id: str) -> dict[str, Any] | None:
    """Return the job's progress snapshot, or None if unknown."""
    value = progress_store.get(_job_key(job_id))
    return value if isinstance(value, dict) else None


def delete_progress(job_id: str, *, commit: bool = True) -> bool:
    """Remove a job's progress snapshot and cancel flag. Returns True if anything was removed."""
    del commit
    with _WRITE_LOCKS[job_id]:
        removed_progress = progress_store.pop(_job_key(job_id), None) is not None
        removed_cancel = progress_store.pop(_cancel_key(job_id), None) is not None
    _WRITE_LOCKS.pop(job_id, None)
    return removed_progress or removed_cancel


def request_cancellation(job_id: str) -> None:
    """Flag a job as cancelled. The GPU loop stops at the next image boundary."""
    progress_store.put(_cancel_key(job_id), _now_iso())


def is_cancelled(job_id: str, *, reload: bool = True) -> bool:
    """Whether cancellation was requested for a job. ``reload`` is kept for compatibility."""
    del reload
    try:
        return progress_store.contains(_cancel_key(job_id))
    except Exception:
        return False


def cleanup_stale_progress_files(*, max_age_hours: int = 24) -> int:
    """Remove progress snapshots not updated for ``max_age_hours``.

    Safety net for crashed jobs; the Dict also expires idle entries after 7 days.
    Returns the number of jobs removed.
    """
    cutoff = datetime.now(timezone.utc).timestamp() - max_age_hours * 3600
    stale: list[str] = []
    for key in list(progress_store.keys()):
        if not isinstance(key, str) or not key.startswith("job:"):
            continue
        value = progress_store.get(key)
        updated = value.get("updated_at") if isinstance(value, dict) else None
        try:
            ts = datetime.fromisoformat(updated).timestamp() if isinstance(updated, str) else 0.0
        except ValueError:
            ts = 0.0
        if ts < cutoff:
            stale.append(key[len("job:"):])

    for safe_id in stale:
        progress_store.pop(f"job:{safe_id}", None)
        progress_store.pop(f"cancel:{safe_id}", None)
    return len(stale)

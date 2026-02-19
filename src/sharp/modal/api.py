"""Modal FastAPI endpoint for SHARP inference and Supabase/R2 upload."""

from __future__ import annotations

import concurrent.futures
import logging
import mimetypes
import os
import re
import shutil
import threading
import uuid
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Sequence
from urllib.parse import quote

import modal
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse

R2_SIGNATURE_VERSION = "s3v4"

logger = logging.getLogger("sharp.modal.api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Updated App Name
APP_NAME = "ml-sharp-optimized"

# Secret names expected to be provisioned in Modal
API_AUTH_SECRET_NAME = "sharp-api-auth"

API_KEY_HEADER = "X-API-KEY"
API_KEY_ENV = "API_AUTH_TOKEN"

DEFAULT_BUCKET = "3dgs-assets"
DEFAULT_EXPORT_FORMATS: Sequence[str] = ("sog",)
DEFAULT_RESULT_TTL_SECONDS = 60 * 60

# Local repo path (resolved at deploy time on your machine, not in container)
REMOTE_REPO_PATH = "/root/ml-sharp"
REMOTE_SRC_PATH = f"{REMOTE_REPO_PATH}/src"

# ROBUST ROOT DETECTION: Works for Local Dev and GitHub Actions
def get_repo_root() -> Path | None:
    if os.environ.get("MODAL_IS_REMOTE") == "1":
        return None

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    return Path.cwd()


REPO_ROOT = get_repo_root()

# Build a Modal image with SHARP + API dependencies
api_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "timm>=1.0.0",
        "scipy>=1.11.0",
        "plyfile>=1.0.0",
        "imageio>=2.30.0",
        "pillow-heif>=0.16.0",
        "numpy>=1.24.0",
        "click>=8.0.0",
        "fastapi",
        "python-multipart",
        "supabase",
        "boto3",
    )
    .run_commands("pip install gsplat --no-build-isolation")
    .env({"PYTHONPATH": REMOTE_SRC_PATH, "MODAL_IS_REMOTE": "1"})
    .add_local_dir(
        str(REPO_ROOT) if REPO_ROOT else "/dummy",
        REMOTE_REPO_PATH,
        ignore=[
            ".git",
            ".venv",
            "__pycache__",
            "node_modules",
            ".conda",
            "venv",
            "output",
        ],
    )
)

# Import volume/constants from app.py at deploy time (runs locally)
from sharp.modal.app import MODEL_CACHE_PATH, TIMEOUT_SECONDS, JobCancelledError, model_volume
from sharp.modal.progress import (
    PROGRESS_VOLUME_PATH,
    cleanup_stale_progress_files,
    delete_progress,
    progress_volume,
    read_progress,
    request_cancellation,
    write_status,
)

# Separate Modal app for the HTTP endpoint
app = modal.App(name=APP_NAME)

_WORKER_KWARGS = dict(
    volumes={MODEL_CACHE_PATH: model_volume, PROGRESS_VOLUME_PATH: progress_volume},
    timeout=TIMEOUT_SECONDS,
    image=api_image,
    secrets=[
        modal.Secret.from_name(API_AUTH_SECRET_NAME),
    ],
)

_MAINTENANCE_KWARGS = dict(
    volumes={PROGRESS_VOLUME_PATH: progress_volume},
    timeout=300,
    image=api_image,
)


def _parse_access_string(access_string: str) -> dict[str, str]:
    """Parse an access string into a dict.

    Supports JSON objects or key=value pairs delimited by ';', ',' or newlines.
    Example: "supabaseUrl=...;supabaseKey=...;supabaseBucket=..."
    """
    import json

    if not access_string or not access_string.strip():
        return {}

    raw = access_string.strip()
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except json.JSONDecodeError:
            pass

    parts: list[str] = []
    for sep in (";", ",", "\n"):
        if sep in raw:
            parts = [p for p in raw.replace("\r", "").split(sep) if p.strip()]
            break
    if not parts:
        parts = [raw]

    parsed: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            parsed[key] = value
    return parsed


def _run_batch_impl(
    image_batch: list[tuple[bytes, str]],
    export_formats: Sequence[str],
    job_id: str | None,
    on_output=None,
    on_error=None,
) -> list[tuple[str, bytes]]:
    from sharp.modal.app import _predict_batch_impl

    return _predict_batch_impl(
        image_batch=image_batch,
        export_formats=tuple(export_formats),
        job_id=job_id,
        on_output=on_output,
        on_error=on_error,
    )


@app.function(gpu="a10", **_WORKER_KWARGS)
def _process_job_on_a10(
    image_batch: list[tuple[bytes, str]],
    export_formats: Sequence[str],
    job_id: str,
    storage_target: str,
    access: dict[str, str],
    prefix: str,
    gpu_request: str,
    ttl_seconds: int,
    results_url: str,
    download_base_url: str,
):
    return _process_job_impl(
        image_batch=image_batch,
        export_formats=export_formats,
        job_id=job_id,
        storage_target=storage_target,
        access=access,
        prefix=prefix,
        gpu_request=gpu_request,
        ttl_seconds=ttl_seconds,
        results_url=results_url,
        download_base_url=download_base_url,
    )


@app.function(gpu="t4", **_WORKER_KWARGS)
def _process_job_on_t4(
    image_batch: list[tuple[bytes, str]],
    export_formats: Sequence[str],
    job_id: str,
    storage_target: str,
    access: dict[str, str],
    prefix: str,
    gpu_request: str,
    ttl_seconds: int,
    results_url: str,
    download_base_url: str,
):
    return _process_job_impl(
        image_batch=image_batch,
        export_formats=export_formats,
        job_id=job_id,
        storage_target=storage_target,
        access=access,
        prefix=prefix,
        gpu_request=gpu_request,
        ttl_seconds=ttl_seconds,
        results_url=results_url,
        download_base_url=download_base_url,
    )


@app.function(gpu="l4", **_WORKER_KWARGS)
def _process_job_on_l4(
    image_batch: list[tuple[bytes, str]],
    export_formats: Sequence[str],
    job_id: str,
    storage_target: str,
    access: dict[str, str],
    prefix: str,
    gpu_request: str,
    ttl_seconds: int,
    results_url: str,
    download_base_url: str,
):
    return _process_job_impl(
        image_batch=image_batch,
        export_formats=export_formats,
        job_id=job_id,
        storage_target=storage_target,
        access=access,
        prefix=prefix,
        gpu_request=gpu_request,
        ttl_seconds=ttl_seconds,
        results_url=results_url,
        download_base_url=download_base_url,
    )


@app.function(gpu="a100", **_WORKER_KWARGS)
def _process_job_on_a100(
    image_batch: list[tuple[bytes, str]],
    export_formats: Sequence[str],
    job_id: str,
    storage_target: str,
    access: dict[str, str],
    prefix: str,
    gpu_request: str,
    ttl_seconds: int,
    results_url: str,
    download_base_url: str,
):
    return _process_job_impl(
        image_batch=image_batch,
        export_formats=export_formats,
        job_id=job_id,
        storage_target=storage_target,
        access=access,
        prefix=prefix,
        gpu_request=gpu_request,
        ttl_seconds=ttl_seconds,
        results_url=results_url,
        download_base_url=download_base_url,
    )


@app.function(gpu="h100", **_WORKER_KWARGS)
def _process_job_on_h100(
    image_batch: list[tuple[bytes, str]],
    export_formats: Sequence[str],
    job_id: str,
    storage_target: str,
    access: dict[str, str],
    prefix: str,
    gpu_request: str,
    ttl_seconds: int,
    results_url: str,
    download_base_url: str,
):
    return _process_job_impl(
        image_batch=image_batch,
        export_formats=export_formats,
        job_id=job_id,
        storage_target=storage_target,
        access=access,
        prefix=prefix,
        gpu_request=gpu_request,
        ttl_seconds=ttl_seconds,
        results_url=results_url,
        download_base_url=download_base_url,
    )


_GPU_JOB_DISPATCH = {
    "a10": _process_job_on_a10,
    "t4": _process_job_on_t4,
    "l4": _process_job_on_l4,
    "a100": _process_job_on_a100,
    "h100": _process_job_on_h100,
}

_SAFE_JOB_ID = re.compile(r"[^a-zA-Z0-9._-]+")
_RESULTS_ROOT = Path(PROGRESS_VOLUME_PATH) / "results"

# Per-job lock for serializing manifest reads/writes across upload threads
_MANIFEST_LOCKS: dict[str, threading.Lock] = {}
_MANIFEST_LOCKS_GUARD = threading.Lock()

# Global lock for volume commits (shared with progress.py's _COMMIT_LOCK pattern)
_VOLUME_COMMIT_LOCK = threading.Lock()

_UPSERT_MAX_RETRIES = 3
_UPSERT_RETRY_DELAY = 0.4

# Seconds to wait after job completion before cleaning up temp results.
# Gives the frontend time to finish downloading before files are removed.
_DEFERRED_CLEANUP_DELAY_SECONDS = 300


def _get_manifest_lock(job_id: str) -> threading.Lock:
    """Get or create a per-job lock for manifest operations."""
    with _MANIFEST_LOCKS_GUARD:
        if job_id not in _MANIFEST_LOCKS:
            _MANIFEST_LOCKS[job_id] = threading.Lock()
        return _MANIFEST_LOCKS[job_id]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _sanitize_job_id(job_id: str) -> str:
    if not job_id:
        raise ValueError("job_id is required")
    return _SAFE_JOB_ID.sub("_", job_id)


def _result_dir(job_id: str) -> Path:
    return _RESULTS_ROOT / _sanitize_job_id(job_id)


def _manifest_path(job_id: str) -> Path:
    return _result_dir(job_id) / "manifest.json"


def _endpoint_url_from_submit_url(submit_url: str, endpoint_slug: str) -> str:
    """Build endpoint URL for Modal unique endpoint hosts.

    Converts:
      https://<user>--<app>-process-image.modal.run
    into:
      https://<user>--<app>-<endpoint_slug>.modal.run
    """
    base = submit_url.strip().rstrip("/")
    needle = "-process-image.modal.run"
    if base.endswith(needle):
        return f"{base[:-len(needle)]}-{endpoint_slug}.modal.run"
    return base


def _build_endpoint_urls(request: Request, job_id: str) -> dict[str, str]:
    submit_base = str(request.base_url).rstrip("/")
    job_id_q = quote(job_id, safe="")

    status_base = _endpoint_url_from_submit_url(submit_base, "get-progress")
    results_base = _endpoint_url_from_submit_url(submit_base, "get-results")
    download_base = _endpoint_url_from_submit_url(submit_base, "download-result")

    return {
        "submit_url": submit_base,
        "status_url": f"{status_base}/?job_id={job_id_q}",
        "results_url": f"{results_base}/?job_id={job_id_q}",
        "download_base_url": download_base,
    }


def _content_disposition_attachment(filename: str) -> str:
    """Build RFC 5987-compatible Content-Disposition value.

    Includes an ASCII fallback (`filename`) and UTF-8 variant (`filename*`)
    to avoid latin-1 header encoding failures for non-ASCII filenames.
    """
    ascii_name = filename.encode("ascii", "ignore").decode("ascii").strip() or "download.bin"
    ascii_name = (
        ascii_name.replace("\\", "_")
        .replace('"', "_")
        .replace("\r", " ")
        .replace("\n", " ")
    )

    utf8_name = quote(filename, safe="")
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{utf8_name}"


def _create_manifest(job_id: str, ttl_seconds: int) -> dict:
    import json

    result_dir = _result_dir(job_id)
    result_dir.mkdir(parents=True, exist_ok=True)

    now = _now_utc()
    manifest = {
        "job_id": job_id,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=max(60, ttl_seconds))).isoformat(),
        "files": [],
    }
    _manifest_path(job_id).write_text(json.dumps(manifest), encoding="utf-8")
    with _VOLUME_COMMIT_LOCK:
        progress_volume.commit()
    return manifest


def _upsert_temp_output(
    job_id: str,
    output_name: str,
    file_bytes: bytes,
    download_base_url: str,
    ttl_seconds: int,
) -> dict:
    """Write a file to temp storage and update the manifest.

    Thread-safe: serialized per-job via _MANIFEST_LOCKS, with retry
    logic for transient volume errors.
    """
    import json
    import time

    filename = Path(output_name).name
    job_id_q = quote(job_id, safe="")
    filename_q = quote(filename, safe="")
    file_entry = {
        "name": filename,
        "size": len(file_bytes),
        "content_type": mimetypes.guess_type(filename)[0] or "application/octet-stream",
        "download_url": f"{download_base_url}/?job_id={job_id_q}&name={filename_q}",
    }

    lock = _get_manifest_lock(job_id)

    for attempt in range(_UPSERT_MAX_RETRIES):
        try:
            with lock:
                result_dir = _result_dir(job_id)
                result_dir.mkdir(parents=True, exist_ok=True)

                # Write the file bytes first
                file_path = result_dir / filename
                file_path.write_bytes(file_bytes)

                # Read existing manifest or create one
                manifest_path = _manifest_path(job_id)
                manifest = None
                if manifest_path.exists():
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        manifest = None
                if not isinstance(manifest, dict):
                    manifest = _create_manifest(job_id, ttl_seconds)

                # Update manifest file list
                files = manifest.get("files", [])
                if not isinstance(files, list):
                    files = []
                files = [item for item in files if not (isinstance(item, dict) and item.get("name") == filename)]
                files.append(file_entry)
                manifest["files"] = files

                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            # Commit outside the manifest lock to avoid holding it during I/O
            with _VOLUME_COMMIT_LOCK:
                progress_volume.commit()

            return file_entry

        except (PermissionError, OSError) as exc:
            if attempt < _UPSERT_MAX_RETRIES - 1:
                logger.warning(
                    "_upsert_temp_output attempt %d failed for %s: %s",
                    attempt + 1, output_name, exc,
                )
                time.sleep(_UPSERT_RETRY_DELAY * (attempt + 1))
                try:
                    progress_volume.reload()
                except Exception:
                    pass
            else:
                raise

    return file_entry  # unreachable


def _load_manifest(job_id: str, *, reload: bool = True) -> dict | None:
    """Load the manifest for a job.

    Args:
        job_id: Job identifier.
        reload: Whether to reload the volume first. Set False when the caller
                already holds a lock or has just committed.
    """
    import json

    if reload:
        progress_volume.reload()

    path = _manifest_path(job_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _manifest_is_expired(manifest: dict) -> bool:
    expires_raw = manifest.get("expires_at")
    if not isinstance(expires_raw, str):
        return True
    try:
        expires_at = datetime.fromisoformat(expires_raw)
    except ValueError:
        return True
    return expires_at <= _now_utc()


def _remove_job_results(job_id: str) -> None:
    result_dir = _result_dir(job_id)
    if result_dir.exists():
        shutil.rmtree(result_dir)
        with _VOLUME_COMMIT_LOCK:
            progress_volume.commit()
    # Clean up manifest lock to prevent unbounded growth
    with _MANIFEST_LOCKS_GUARD:
        _MANIFEST_LOCKS.pop(job_id, None)


def _cleanup_job_results(job_id: str) -> None:
    """Remove all temp result files, manifest, and progress for a job.

    Idempotent — safe to call multiple times or if already cleaned.
    """
    result_dir = _result_dir(job_id)
    removed = False

    try:
        if result_dir.exists():
            shutil.rmtree(result_dir, ignore_errors=True)
            removed = True
    except OSError:
        logger.warning("Failed to rmtree results for %s", job_id, exc_info=True)

    try:
        delete_progress(job_id, commit=False)
    except Exception:
        logger.warning("Failed to delete progress for %s", job_id, exc_info=True)

    if removed:
        try:
            with _VOLUME_COMMIT_LOCK:
                progress_volume.commit()
        except Exception:
            logger.warning("Failed to commit after cleanup for %s", job_id, exc_info=True)

    # Clean up the manifest lock
    with _MANIFEST_LOCKS_GUARD:
        _MANIFEST_LOCKS.pop(job_id, None)

    logger.info("Cleaned up job %s (results_removed=%s)", job_id, removed)


def _cleanup_expired_results() -> int:
    """Remove expired or orphaned job result directories.

    Handles:
    - Jobs with valid manifests whose expires_at has passed
    - Jobs with missing or corrupt manifests (age-based fallback)
    - Orphaned directories with no manifest
    Also cleans up each job's progress file.
    """
    import json

    progress_volume.reload()
    if not _RESULTS_ROOT.exists():
        return 0

    now = _now_utc()
    now_ts = now.timestamp()
    removed = 0

    for job_dir in _RESULTS_ROOT.iterdir():
        if not job_dir.is_dir():
            continue

        job_id = job_dir.name
        manifest_path = job_dir / "manifest.json"
        should_remove = False

        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expires_raw = manifest.get("expires_at")
                if isinstance(expires_raw, str):
                    expires_at = datetime.fromisoformat(expires_raw)
                    if now >= expires_at:
                        should_remove = True
                else:
                    # No expires_at field — fall back to directory age
                    dir_age_hours = (now_ts - job_dir.stat().st_mtime) / 3600
                    if dir_age_hours > 1:
                        should_remove = True
            except Exception:
                # Corrupt manifest — remove if old enough
                try:
                    dir_age_hours = (now_ts - job_dir.stat().st_mtime) / 3600
                    if dir_age_hours > 1:
                        should_remove = True
                except OSError:
                    should_remove = True
        else:
            # No manifest at all — orphaned directory, check age
            try:
                dir_age_hours = (now_ts - job_dir.stat().st_mtime) / 3600
                if dir_age_hours > 1:
                    should_remove = True
            except OSError:
                should_remove = True

        if should_remove:
            try:
                shutil.rmtree(job_dir, ignore_errors=True)
                removed += 1
                # Also clean up the progress file
                try:
                    delete_progress(job_id, commit=False)
                except Exception:
                    pass
            except Exception:
                logger.warning("Failed to remove %s", job_dir, exc_info=True)

    if removed:
        try:
            with _VOLUME_COMMIT_LOCK:
                progress_volume.commit()
        except Exception:
            logger.warning("Failed to commit after cron cleanup", exc_info=True)

    return removed


@app.function(
    schedule=modal.Period(days=1),
    **_MAINTENANCE_KWARGS,
)
def cleanup_expired_results_job() -> dict[str, int]:
    """Periodic cleanup for expired temporary result files and orphaned progress files."""
    removed = _cleanup_expired_results()
    if removed:
        logger.info("Removed %d expired result job(s)", removed)

    # Also clean up orphaned progress files (e.g., from crashed jobs)
    stale_count = cleanup_stale_progress_files(max_age_hours=24)
    if stale_count:
        logger.info("Cleaned up %d stale progress files", stale_count)

    return {
        "expired_dirs_removed": removed,
        "stale_progress_removed": stale_count,
    }


def _authorize_request(request: Request) -> Response | None:
    api_key = os.environ.get(API_KEY_ENV)
    if api_key and request.headers.get(API_KEY_HEADER) != api_key:
        return Response(status_code=401)
    return None


def _upload_to_r2(*, file_content: bytes, filename: str, config: dict[str, str]) -> str:
    """Upload file to Cloudflare R2 using S3-compatible API."""
    import mimetypes

    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError

    endpoint = config["s3Endpoint"]
    bucket = config["s3Bucket"]
    prefix = config.get("prefix") or ""
    full_key = f"{prefix}/{filename}".strip("/")
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    logger.info(
        "R2 _upload_to_r2: endpoint=%s bucket=%s key=%s content_type=%s",
        endpoint,
        bucket,
        full_key,
        content_type,
    )

    try:
        s3 = boto3.client(
            service_name="s3",
            endpoint_url=endpoint,
            aws_access_key_id=config["s3AccessKeyId"],
            aws_secret_access_key=config["s3SecretAccessKey"],
            region_name="auto",
            config=Config(signature_version=R2_SIGNATURE_VERSION),
        )

        resp = s3.put_object(
            Bucket=bucket,
            Key=full_key,
            Body=file_content,
            ContentType=content_type,
        )
        logger.info("R2 put_object response: status=%s etag=%s", resp.get("ResponseMetadata", {}).get("HTTPStatusCode"), resp.get("ETag"))
    except ClientError as e:
        logger.exception("R2 ClientError: %s", e.response.get("Error", {}))
        raise

    # Construct public URL - use custom base if provided, else default R2 pattern
    public_base = config.get("s3PublicUrlBase")
    if public_base:
        # Custom domain: e.g. https://assets.example.com
        url = f"{public_base.rstrip('/')}/{full_key}"
    else:
        # Default R2 public URL pattern (requires R2.dev subdomain enabled or custom domain)
        # Note: This may not work if public access isn't configured on the bucket
        url = f"https://{bucket}.r2.cloudflarestorage.com/{full_key}"
        logger.warning(
            "No s3PublicUrlBase provided; using default pattern which may not be publicly accessible: %s",
            url,
        )

    return url


def _expected_file_count(image_batch: list[tuple[bytes, str]], export_formats: Sequence[str]) -> int:
    unique_formats = tuple(dict.fromkeys(fmt.lower() for fmt in export_formats if fmt))
    normalized = unique_formats or DEFAULT_EXPORT_FORMATS
    return len(image_batch) * len(normalized)


def _update_incremental_progress(
    job_id: str,
    *,
    total_steps: int,
    files_expected: int,
    files: list[dict[str, str]],
    file_errors: list[dict[str, str]],
    phase: str,
    message: str,
    step: int,
    done: bool,
    status: str = "running",
    result_type: str,
    expires_at: str | None = None,
    results_url: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": status,
        "phase": phase,
        "message": message,
        "step": step,
        "total_steps": total_steps,
        "done": done,
        "result_type": result_type,
        "files": files,
        "result_files": files,
        "files_ready": len(files),
        "files_expected": files_expected,
        "file_errors": file_errors,
    }
    if expires_at:
        payload["expires_at"] = expires_at
    if results_url:
        payload["results_url"] = results_url
        payload["results_path"] = results_url
    return write_status(job_id, **payload)


def _process_job_impl(
    image_batch: list[tuple[bytes, str]],
    export_formats: Sequence[str],
    job_id: str,
    storage_target: str,
    access: dict[str, str],
    prefix: str,
    gpu_request: str,
    ttl_seconds: int,
    results_url: str,
    download_base_url: str,
) -> dict:
    total_steps = len(image_batch)
    files_expected = _expected_file_count(image_batch, export_formats)

    write_status(
        job_id,
        status="running",
        phase="starting_worker",
        step=0,
        total_steps=total_steps,
        message=f"Worker started on {gpu_request.upper()}",
        done=False,
        files=[],
        result_files=[],
        files_ready=0,
        files_expected=files_expected,
        file_errors=[],
    )

    result_type = "cloud" if storage_target in {"r2", "supabase"} else "temp"
    if result_type == "temp":
        manifest = _create_manifest(job_id, ttl_seconds)
        expires_at = manifest.get("expires_at") if isinstance(manifest, dict) else None
    else:
        expires_at = None

    files: list[dict[str, str]] = []
    file_errors: list[dict[str, str]] = []
    state_lock = threading.Lock()

    # Use a thread pool for concurrent uploads so the GPU thread is never blocked.
    # max_workers=3 gives good upload throughput without overwhelming the endpoint.
    upload_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=3, thread_name_prefix=f"upload-{job_id[:8]}"
    )
    upload_futures: list[concurrent.futures.Future] = []

    if storage_target == "r2":
        required = ("s3Endpoint", "s3AccessKeyId", "s3SecretAccessKey", "s3Bucket")
        missing = [key for key in required if not access.get(key)]
        if missing:
            raise ValueError(f"storageTarget=r2 requires: {', '.join(missing)}")
        r2_config = {
            "s3Endpoint": access["s3Endpoint"],
            "s3AccessKeyId": access["s3AccessKeyId"],
            "s3SecretAccessKey": access["s3SecretAccessKey"],
            "s3Bucket": access["s3Bucket"],
            "prefix": prefix,
            "s3PublicUrlBase": access.get("s3PublicUrlBase"),
        }
    else:
        r2_config = {}

    if storage_target == "supabase":
        supabase_url = access.get("supabaseUrl") or access.get("SUPABASE_URL")
        supabase_key = access.get("supabaseKey") or access.get("SUPABASE_KEY")
        bucket = access.get("supabaseBucket") or access.get("SUPABASE_BUCKET") or DEFAULT_BUCKET
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase accessString missing supabaseUrl/supabaseKey")

        from supabase import create_client

        supabase_client = create_client(supabase_url, supabase_key)
        supabase_prefix = prefix or "collections/default/assets"
    else:
        supabase_client = None
        bucket = ""
        supabase_prefix = ""

    def _state_snapshot() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        with state_lock:
            return list(files), list(file_errors)

    def _write_progress(
        *,
        phase: str,
        message: str,
        step: int,
        done: bool,
        status: str = "running",
    ) -> None:
        current_files, current_errors = _state_snapshot()
        _update_incremental_progress(
            job_id,
            total_steps=total_steps,
            files_expected=files_expected,
            files=current_files,
            file_errors=current_errors,
            phase=phase,
            message=message,
            step=step,
            done=done,
            status=status,
            result_type=result_type,
            expires_at=expires_at if isinstance(expires_at, str) else None,
            results_url=results_url if result_type == "temp" else None,
        )

    def _deliver_output(
        output_name: str, file_bytes: bytes, image_index: int, total_images: int
    ) -> None:
        """Upload/stage a single output file. Runs on a background thread."""
        try:
            if storage_target == "r2":
                url = _upload_to_r2(file_content=file_bytes, filename=output_name, config=r2_config)
                object_key = f"{prefix}/{output_name}".strip("/")
                file_entry = {"name": output_name, "path": object_key, "url": url}
            elif storage_target == "supabase":
                object_key = str(Path(supabase_prefix) / output_name)
                file_handle = BytesIO(file_bytes)
                assert supabase_client is not None
                supabase_client.storage.from_(bucket).upload(
                    object_key,
                    file_handle.getvalue(),
                    {
                        "content-type": "application/octet-stream",
                        "upsert": "true",
                    },
                )
                url = supabase_client.storage.from_(bucket).get_public_url(object_key)
                file_entry = {"name": output_name, "path": object_key, "url": url}
            else:
                file_entry = _upsert_temp_output(
                    job_id=job_id,
                    output_name=output_name,
                    file_bytes=file_bytes,
                    download_base_url=download_base_url,
                    ttl_seconds=ttl_seconds,
                )

            with state_lock:
                files[:] = [entry for entry in files if entry.get("name") != file_entry.get("name")]
                files.append(file_entry)

            _write_progress(
                phase="uploading_or_staging_results",
                message=f"Ready: {output_name} ({image_index}/{total_images})",
                step=image_index,
                done=False,
            )
        except Exception as exc:
            error_message = f"{output_name}: {exc}"
            logger.exception("Failed delivering output %s", output_name)
            with state_lock:
                file_errors.append({"name": output_name, "error": error_message})
            _write_progress(
                phase="uploading_or_staging_results",
                message=f"Failed delivering {output_name}: {exc}",
                step=image_index,
                done=False,
            )

    def _on_output(output_name: str, file_bytes: bytes, image_index: int, total_images: int) -> None:
        """Submit upload work to the thread pool — returns immediately, never blocks GPU."""
        future = upload_executor.submit(
            _deliver_output, output_name, file_bytes, image_index, total_images
        )
        upload_futures.append(future)

    def _on_error(item_name: str, error_message: str, image_index: int, total_images: int) -> None:
        with state_lock:
            file_errors.append({"name": item_name, "error": error_message})
        _write_progress(
            phase="processing_images",
            message=f"Skipped {item_name}: {error_message}",
            step=image_index,
            done=False,
        )

    failure: Exception | None = None
    try:
        write_status(
            job_id,
            status="running",
            phase="dispatching_inference",
            step=0,
            total_steps=total_steps,
            message="Dispatching inference",
            done=False,
            files=files,
            result_files=files,
            files_ready=0,
            files_expected=files_expected,
            file_errors=file_errors,
        )
        _run_batch_impl(
            image_batch=image_batch,
            export_formats=export_formats,
            job_id=job_id,
            on_output=_on_output,
            on_error=_on_error,
        )
    except Exception as exc:
        failure = exc
    finally:
        # Wait for all in-flight uploads to finish before finalizing.
        # Each future's exception is already handled inside _deliver_output,
        # but we still call .result() to surface unexpected errors.
        for fut in concurrent.futures.as_completed(upload_futures):
            try:
                fut.result()
            except Exception as exc:
                logger.error("Unexpected upload error: %s", exc, exc_info=True)
        upload_executor.shutdown(wait=True)

    if failure is not None:
        is_cancel = isinstance(failure, JobCancelledError)
        if is_cancel:
            _write_progress(
                phase="cancelled",
                message="Job cancelled by user",
                step=0,
                done=True,
                status="cancelled",
            )
            # Immediately clean up temp files on cancel
            _cleanup_job_results(job_id)
            return {
                "job_id": job_id,
                "result_type": result_type,
                "status": "cancelled",
                "files": [],
                "file_errors": [],
            }

        _write_progress(
            phase="failed",
            message=f"Failed: {failure}",
            step=0,
            done=True,
            status="failed",
        )
        raise failure

    completion_message = "Processing completed"
    if file_errors:
        completion_message = f"Completed with {len(file_errors)} file error(s)"

    _write_progress(
        phase="completed",
        message=completion_message,
        step=total_steps,
        done=True,
        status="complete",
    )

    files_snapshot, file_errors_snapshot = _state_snapshot()

    result: dict[str, object] = {
        "job_id": job_id,
        "result_type": result_type,
        "files": files_snapshot,
        "file_errors": file_errors_snapshot,
    }
    if result_type == "temp":
        result["results_url"] = results_url
        result["expires_at"] = expires_at

    # Clean up the progress file now that the job is fully done.
    # Results are tracked via the manifest in _RESULTS_ROOT; the
    # top-level progress JSON is no longer needed.
    # NOTE: Do NOT delete_progress here — the frontend still needs it
    # to discover download URLs. The cron job cleans it up after 24h.

    # Schedule deferred cleanup — gives the frontend time to download,
    # then removes temp files regardless of whether consume=true was used.
    if result_type == "temp":
        import time as _time

        def _deferred_cleanup() -> None:
            _time.sleep(_DEFERRED_CLEANUP_DELAY_SECONDS)
            logger.info("Running deferred cleanup for job %s", job_id)
            _cleanup_job_results(job_id)

        cleanup_thread = threading.Thread(
            target=_deferred_cleanup,
            name=f"cleanup-{job_id[:8]}",
            daemon=True,
        )
        cleanup_thread.start()

    return result


@app.function(
    **_WORKER_KWARGS,
)
def _process_job(
    image_batch: list[tuple[bytes, str]],
    export_formats: Sequence[str],
    job_id: str,
    storage_target: str,
    access: dict[str, str],
    prefix: str,
    gpu_request: str,
    ttl_seconds: int,
    results_url: str,
    download_base_url: str,
) -> dict:
    """Background job processor for async API flow."""
    return _process_job_impl(
        image_batch=image_batch,
        export_formats=export_formats,
        job_id=job_id,
        storage_target=storage_target,
        access=access,
        prefix=prefix,
        gpu_request=gpu_request,
        ttl_seconds=ttl_seconds,
        results_url=results_url,
        download_base_url=download_base_url,
    )


@app.function(
    **_WORKER_KWARGS,
)
@modal.fastapi_endpoint(method="POST")
async def process_image(request: Request):
    """Submit SHARP job and return immediately with a pollable job id."""

    unauthorized = _authorize_request(request)
    if unauthorized:
        return unauthorized

    form = await request.form()
    uploads = form.getlist("file") or form.getlist("files")
    if not uploads:
        raise HTTPException(
            status_code=400,
            detail="Form field 'file' (or 'files') is required.",
        )

    image_batch: list[tuple[bytes, str]] = []
    for upload in uploads:
        filename = upload.filename or "upload.png"
        image_bytes = await upload.read()
        image_batch.append((image_bytes, filename))

    job_id = (
        form.get("jobId")
        or form.get("job_id")
        or f"job-{uuid.uuid4().hex}"
    )
    job_id = _sanitize_job_id(str(job_id))
    endpoint_urls = _build_endpoint_urls(request, job_id)
    total_steps = len(image_batch)
    write_status(
        job_id,
        status="queued",
        phase="queued",
        step=0,
        total_steps=total_steps,
        message="Queued",
        done=False,
        files=[],
        result_files=[],
        files_ready=0,
        files_expected=0,
        file_errors=[],
    )

    formats_raw = form.get("format") or form.get("formats")
    export_formats: Sequence[str]
    if isinstance(formats_raw, str) and formats_raw.strip():
        export_formats = [fmt.strip() for fmt in formats_raw.split(",") if fmt.strip()]
    else:
        export_formats = DEFAULT_EXPORT_FORMATS

    # Explicit storage target: "r2" | "supabase" | anything else -> temp result storage
    storage_target = (form.get("storageTarget") or "").strip().lower()
    prefix = form.get("prefix") or ""
    logger.info("Received storageTarget=%r, prefix=%r", storage_target, prefix)

    access_string = form.get("accessString") or form.get("access") or ""
    access = _parse_access_string(access_string)

    # Pre-validate storage requirements before GPU work
    if storage_target == "r2":
        required = ("s3Endpoint", "s3AccessKeyId", "s3SecretAccessKey", "s3Bucket")
        missing = [key for key in required if not access.get(key)]
        if missing:
            logger.warning("R2 selected but missing fields: %s", missing)
            write_status(
                job_id,
                status="failed",
                phase="validation_failed",
                step=0,
                total_steps=total_steps,
                message=f"Missing fields: {', '.join(missing)}",
                done=True,
                error=f"storageTarget=r2 requires: {', '.join(missing)}",
            )
            raise HTTPException(
                status_code=400,
                detail=f"storageTarget=r2 requires: {', '.join(missing)}",
            )
    elif storage_target == "supabase":
        missing = []
        if not (access.get("supabaseUrl") or access.get("SUPABASE_URL")):
            missing.append("supabaseUrl")
        if not (access.get("supabaseKey") or access.get("SUPABASE_KEY")):
            missing.append("supabaseKey")
        if missing:
            logger.warning("Supabase selected but missing fields: %s", missing)
            write_status(
                job_id,
                status="failed",
                phase="validation_failed",
                step=0,
                total_steps=total_steps,
                message=f"Missing fields: {', '.join(missing)}",
                done=True,
                error=f"storageTarget=supabase requires: {', '.join(missing)}",
            )
            raise HTTPException(
                status_code=400,
                detail=f"storageTarget=supabase requires: {', '.join(missing)}",
            )

    gpu_request = (form.get("gpu") or form.get("gpu_type") or "a10").strip().lower()
    if gpu_request not in _GPU_JOB_DISPATCH and gpu_request not in {"cpu", "none"}:
        write_status(
            job_id,
            status="failed",
            phase="validation_failed",
            step=0,
            total_steps=total_steps,
            message=f"Unsupported gpu: {gpu_request}",
            done=True,
            error=f"Unsupported gpu: {gpu_request}",
        )
        raise HTTPException(status_code=400, detail=f"Unsupported gpu: {gpu_request}")

    write_status(
        job_id,
        status="queued",
        phase="starting_worker",
        step=0,
        total_steps=total_steps,
        message=f"Starting worker on {gpu_request.upper()}",
        done=False,
        files=[],
        result_files=[],
        files_ready=0,
        files_expected=0,
        file_errors=[],
    )

    ttl_raw = form.get("resultTtlSeconds") or form.get("ttlSeconds")
    try:
        ttl_seconds = int(ttl_raw) if ttl_raw else DEFAULT_RESULT_TTL_SECONDS
    except ValueError:
        raise HTTPException(status_code=400, detail="resultTtlSeconds must be an integer")

    processor = _GPU_JOB_DISPATCH.get(gpu_request, _process_job)
    call = processor.spawn(
        image_batch=image_batch,
        export_formats=export_formats,
        job_id=job_id,
        storage_target=storage_target,
        access=access,
        prefix=prefix,
        gpu_request=gpu_request,
        ttl_seconds=ttl_seconds,
        results_url=endpoint_urls["results_url"],
        download_base_url=endpoint_urls["download_base_url"],
    )

    return JSONResponse(
        status_code=202,
        content={
            "accepted": True,
            "job_id": job_id,
            "status": "queued",
            "submit_url": endpoint_urls["submit_url"],
            "status_url": endpoint_urls["status_url"],
            "results_url": endpoint_urls["results_url"],
            "status_path": endpoint_urls["status_url"],
            "results_path": endpoint_urls["results_url"],
            "call_id": call.object_id,
        },
    )


@app.function(
    **_WORKER_KWARGS,
)
@modal.fastapi_endpoint(method="GET")
def get_progress(request: Request, job_id: str):
    """Return the latest progress snapshot for a job."""
    unauthorized = _authorize_request(request)
    if unauthorized:
        return unauthorized

    progress = read_progress(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found")
    return progress


@app.function(
    **_WORKER_KWARGS,
)
@modal.fastapi_endpoint(method="POST")
def cancel_job(request: Request, job_id: str):
    """Cancel a running job and clean up all related resources.

    Writes a cancel sentinel to the progress volume so the GPU inference
    loop stops at the next image boundary.  Immediately cleans up any
    temp result files and marks the progress as cancelled.
    """
    unauthorized = _authorize_request(request)
    if unauthorized:
        return unauthorized

    if not job_id or not job_id.strip():
        raise HTTPException(status_code=400, detail="job_id is required")

    progress = read_progress(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found")

    # If already done/cancelled, just return current state
    if progress.get("done"):
        return {
            "job_id": job_id,
            "status": progress.get("status", "unknown"),
            "message": "Job already finished",
            "cancelled": progress.get("status") == "cancelled",
        }

    # 1. Write the cancel sentinel so the GPU loop stops
    request_cancellation(job_id)

    # 2. Update progress to cancelled immediately so the frontend sees it
    write_status(
        job_id,
        status="cancelled",
        phase="cancelled",
        message="Job cancelled by user",
        step=progress.get("step", 0),
        total_steps=progress.get("total_steps", 0),
        done=True,
        commit=True,
    )

    # 3. Clean up temp result files immediately
    try:
        _cleanup_job_results(job_id)
    except Exception:
        logger.warning("Failed cleanup during cancel for %s", job_id, exc_info=True)

    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": "Job cancelled and resources cleaned up",
        "cancelled": True,
    }


@app.function(
    **_WORKER_KWARGS,
)
@modal.fastapi_endpoint(method="GET")
def get_results(request: Request, job_id: str):
    """Get result manifest or partial files for a job."""
    unauthorized = _authorize_request(request)
    if unauthorized:
        return unauthorized

    progress = read_progress(job_id) or {}
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found")

    complete = bool(progress.get("done"))
    result_type = str(progress.get("result_type") or "")

    if result_type == "cloud":
        files = progress.get("files", [])
        if not isinstance(files, list):
            files = []
        return {
            "job_id": job_id,
            "complete": complete,
            "result_type": "cloud",
            "files": files,
            "file_errors": progress.get("file_errors", []),
        }

    manifest = _load_manifest(job_id)
    if not manifest:
        files = progress.get("files", [])
        if not isinstance(files, list):
            files = []
        if not files:
            raise HTTPException(status_code=404, detail="Results not found")
        return {
            "job_id": job_id,
            "complete": complete,
            "result_type": "temp",
            "expires_at": progress.get("expires_at"),
            "files": files,
            "file_errors": progress.get("file_errors", []),
        }

    if _manifest_is_expired(manifest):
        _remove_job_results(job_id)
        raise HTTPException(status_code=404, detail="Results expired")

    return {
        "job_id": job_id,
        "complete": complete,
        "result_type": "temp",
        "expires_at": manifest.get("expires_at"),
        "files": manifest.get("files", []),
        "file_errors": progress.get("file_errors", []),
    }


@app.function(
    **_WORKER_KWARGS,
)
@modal.fastapi_endpoint(method="GET")
def download_result(request: Request, job_id: str, name: str, consume: bool = True):
    """Download one temporary result file.

    By default ``consume=true``, which deletes the file after it is read.
    Concurrent downloads are safe: each request locks the manifest while
    mutating it, and file bytes are read before any deletion.
    """
    import json

    unauthorized = _authorize_request(request)
    if unauthorized:
        return unauthorized

    manifest = _load_manifest(job_id)
    if not manifest:
        raise HTTPException(status_code=404, detail="Results not found")
    if _manifest_is_expired(manifest):
        _remove_job_results(job_id)
        raise HTTPException(status_code=404, detail="Results expired")

    target_name = Path(name).name
    files = manifest.get("files", []) if isinstance(manifest, dict) else []
    entry = None
    for item in files:
        if isinstance(item, dict) and item.get("name") == target_name:
            entry = item
            break
    if entry is None:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = _result_dir(job_id) / target_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Read file bytes BEFORE any mutations so concurrent commits can't
    # invalidate the local filesystem view underneath us.
    file_bytes = file_path.read_bytes()
    media_type = entry.get("content_type") if isinstance(entry, dict) else None
    if not isinstance(media_type, str) or not media_type:
        media_type = mimetypes.guess_type(target_name)[0] or "application/octet-stream"

    if consume:
        lock = _get_manifest_lock(job_id)
        with lock:
            # Re-read manifest under lock — another download may have changed it
            current_manifest = _load_manifest(job_id, reload=False) or manifest
            current_files = current_manifest.get("files", [])
            if not isinstance(current_files, list):
                current_files = []

            remaining = [
                item
                for item in current_files
                if isinstance(item, dict) and item.get("name") != target_name
            ]
            current_manifest["files"] = remaining

            file_path.unlink(missing_ok=True)

            if remaining:
                _manifest_path(job_id).write_text(
                    json.dumps(current_manifest), encoding="utf-8"
                )
            else:
                # Last file consumed — remove the entire result directory
                result_dir = _result_dir(job_id)
                if result_dir.exists():
                    shutil.rmtree(result_dir, ignore_errors=True)

        # Commit outside the lock
        with _VOLUME_COMMIT_LOCK:
            progress_volume.commit()

        # Update progress so the frontend sees the updated file list.
        # This is lightweight and non-critical — best-effort.
        try:
            progress = read_progress(job_id) or {}
            files_expected = progress.get("files_expected")
            if not isinstance(files_expected, int):
                files_expected = len(remaining)
            total_steps = progress.get("total_steps")
            if not isinstance(total_steps, int):
                total_steps = 0
            step = progress.get("step")
            if not isinstance(step, int):
                step = total_steps

            write_status(
                job_id,
                status=str(progress.get("status") or "running"),
                phase=str(progress.get("phase") or "uploading_or_staging_results"),
                message=str(progress.get("message") or "Updated result files"),
                step=step,
                total_steps=total_steps,
                done=bool(progress.get("done")),
                result_type=str(progress.get("result_type") or "temp"),
                files=remaining,
                result_files=remaining,
                files_ready=len(remaining),
                files_expected=files_expected,
                file_errors=progress.get("file_errors", []),
                expires_at=current_manifest.get("expires_at"),
                results_url=progress.get("results_url"),
                results_path=progress.get("results_path"),
                commit=False,  # Already committed above
            )
        except Exception:
            logger.warning("Failed updating progress after consume for %s", job_id, exc_info=True)

        # If all files consumed, clean up the progress file too
        if not remaining:
            try:
                delete_progress(job_id, commit=True)
            except Exception:
                pass  # Cron will catch it

    response = Response(content=file_bytes, media_type=media_type)
    response.headers["Content-Disposition"] = _content_disposition_attachment(target_name)
    return response

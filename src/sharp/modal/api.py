"""Modal FastAPI endpoint for SHARP inference and Supabase/R2 upload."""

from __future__ import annotations

import logging
import mimetypes
import os
import re
import shutil
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
from sharp.modal.app import MODEL_CACHE_PATH, TIMEOUT_SECONDS, model_volume
from sharp.modal.progress import PROGRESS_VOLUME_PATH, progress_volume, read_progress, write_status

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
    image_batch: list[tuple[bytes, str]], export_formats: Sequence[str], job_id: str | None
) -> list[tuple[str, bytes]]:
    from sharp.modal.app import _predict_batch_impl

    return _predict_batch_impl(
        image_batch=image_batch,
        export_formats=tuple(export_formats),
        job_id=job_id,
    )


@app.function(gpu="a10", **_WORKER_KWARGS)
def _predict_on_a10(
    image_batch: list[tuple[bytes, str]],
    export_formats: Sequence[str],
    job_id: str | None = None,
):
    return _run_batch_impl(image_batch, export_formats, job_id)


@app.function(gpu="t4", **_WORKER_KWARGS)
def _predict_on_t4(
    image_batch: list[tuple[bytes, str]],
    export_formats: Sequence[str],
    job_id: str | None = None,
):
    return _run_batch_impl(image_batch, export_formats, job_id)


@app.function(gpu="l4", **_WORKER_KWARGS)
def _predict_on_l4(
    image_batch: list[tuple[bytes, str]],
    export_formats: Sequence[str],
    job_id: str | None = None,
):
    return _run_batch_impl(image_batch, export_formats, job_id)


@app.function(gpu="a100", **_WORKER_KWARGS)
def _predict_on_a100(
    image_batch: list[tuple[bytes, str]],
    export_formats: Sequence[str],
    job_id: str | None = None,
):
    return _run_batch_impl(image_batch, export_formats, job_id)


@app.function(gpu="h100", **_WORKER_KWARGS)
def _predict_on_h100(
    image_batch: list[tuple[bytes, str]],
    export_formats: Sequence[str],
    job_id: str | None = None,
):
    return _run_batch_impl(image_batch, export_formats, job_id)


_GPU_DISPATCH = {
    "a10": _predict_on_a10,
    "t4": _predict_on_t4,
    "l4": _predict_on_l4,
    "a100": _predict_on_a100,
    "h100": _predict_on_h100,
}

_SAFE_JOB_ID = re.compile(r"[^a-zA-Z0-9._-]+")
_RESULTS_ROOT = Path(PROGRESS_VOLUME_PATH) / "results"


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


def _save_temp_outputs(
    job_id: str,
    outputs: list[tuple[str, bytes]],
    ttl_seconds: int,
    download_base_url: str,
) -> dict:
    import json

    result_dir = _result_dir(job_id)
    result_dir.mkdir(parents=True, exist_ok=True)

    files: list[dict[str, str | int]] = []
    job_id_q = quote(job_id, safe="")
    for output_name, file_bytes in outputs:
        filename = Path(output_name).name
        filename_q = quote(filename, safe="")
        file_path = result_dir / filename
        file_path.write_bytes(file_bytes)
        files.append(
            {
                "name": filename,
                "size": len(file_bytes),
                "content_type": mimetypes.guess_type(filename)[0] or "application/octet-stream",
                "download_url": f"{download_base_url}/?job_id={job_id_q}&name={filename_q}",
            }
        )

    now = _now_utc()
    manifest = {
        "job_id": job_id,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=max(60, ttl_seconds))).isoformat(),
        "files": files,
    }
    _manifest_path(job_id).write_text(json.dumps(manifest), encoding="utf-8")
    progress_volume.commit()
    return manifest


def _load_manifest(job_id: str) -> dict | None:
    import json

    progress_volume.reload()
    path = _manifest_path(job_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _remove_job_results(job_id: str) -> None:
    result_dir = _result_dir(job_id)
    if result_dir.exists():
        shutil.rmtree(result_dir)
        progress_volume.commit()


def _cleanup_expired_results() -> int:
    import json

    progress_volume.reload()
    if not _RESULTS_ROOT.exists():
        return 0

    now = _now_utc()
    removed = 0
    for job_dir in _RESULTS_ROOT.iterdir():
        if not job_dir.is_dir():
            continue
        manifest_path = job_dir / "manifest.json"
        if not manifest_path.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
            removed += 1
            continue

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expires_raw = manifest.get("expires_at")
            expires_at = datetime.fromisoformat(expires_raw) if isinstance(expires_raw, str) else None
        except Exception:
            expires_at = None

        if expires_at is None or expires_at <= now:
            shutil.rmtree(job_dir, ignore_errors=True)
            removed += 1

    if removed:
        progress_volume.commit()
    return removed


@app.function(
    schedule=modal.Period(minutes=30),
    **_WORKER_KWARGS,
)
def cleanup_expired_results_job() -> dict[str, int]:
    """Periodic cleanup for expired temporary result files."""
    removed = _cleanup_expired_results()
    if removed:
        logger.info("Removed %d expired result job(s)", removed)
    return {"removed": removed}


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
    from sharp.modal.app import _predict_batch_impl

    total_steps = len(image_batch)
    write_status(
        job_id,
        status="running",
        phase="starting_worker",
        step=0,
        total_steps=total_steps,
        message=f"Worker started on {gpu_request.upper()}",
        done=False,
    )

    try:
        write_status(
            job_id,
            status="running",
            phase="dispatching_inference",
            step=0,
            total_steps=total_steps,
            message="Dispatching inference",
            done=False,
        )
        if gpu_request in _GPU_DISPATCH:
            outputs = _GPU_DISPATCH[gpu_request].remote(
                image_batch=image_batch,
                export_formats=export_formats,
                job_id=job_id,
            )
        else:
            outputs = _predict_batch_impl(
                image_batch=image_batch,
                export_formats=tuple(export_formats),
                job_id=job_id,
            )
    except Exception as e:
        write_status(
            job_id,
            status="failed",
            phase="failed",
            step=0,
            total_steps=total_steps,
            message=f"Failed: {e}",
            done=True,
            error=str(e),
        )
        raise

    if storage_target == "r2":
        write_status(
            job_id,
            status="running",
            phase="uploading_or_staging_results",
            step=total_steps,
            total_steps=total_steps,
            message="Uploading outputs to R2",
            done=False,
        )
        required = ("s3Endpoint", "s3AccessKeyId", "s3SecretAccessKey", "s3Bucket")
        missing = [key for key in required if not access.get(key)]
        if missing:
            raise ValueError(f"storageTarget=r2 requires: {', '.join(missing)}")

        uploaded_files: list[dict[str, str]] = []
        r2_config = {
            "s3Endpoint": access["s3Endpoint"],
            "s3AccessKeyId": access["s3AccessKeyId"],
            "s3SecretAccessKey": access["s3SecretAccessKey"],
            "s3Bucket": access["s3Bucket"],
            "prefix": prefix,
            "s3PublicUrlBase": access.get("s3PublicUrlBase"),
        }
        for output_name, file_bytes in outputs:
            url = _upload_to_r2(file_content=file_bytes, filename=output_name, config=r2_config)
            object_key = f"{prefix}/{output_name}".strip("/")
            uploaded_files.append({"name": output_name, "path": object_key, "url": url})

        write_status(
            job_id,
            status="complete",
            phase="completed",
            step=total_steps,
            total_steps=total_steps,
            message="Uploaded outputs to R2",
            done=True,
            result_type="cloud",
            files=uploaded_files,
        )
        return {"job_id": job_id, "files": uploaded_files, "result_type": "cloud"}

    if storage_target == "supabase":
        write_status(
            job_id,
            status="running",
            phase="uploading_or_staging_results",
            step=total_steps,
            total_steps=total_steps,
            message="Uploading outputs to Supabase",
            done=False,
        )
        supabase_url = access.get("supabaseUrl") or access.get("SUPABASE_URL")
        supabase_key = access.get("supabaseKey") or access.get("SUPABASE_KEY")
        bucket = access.get("supabaseBucket") or access.get("SUPABASE_BUCKET") or DEFAULT_BUCKET
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase accessString missing supabaseUrl/supabaseKey")

        from supabase import create_client

        client = create_client(supabase_url, supabase_key)
        upload_prefix = prefix or "collections/default/assets"

        uploaded_files: list[dict[str, str]] = []
        for output_name, file_bytes in outputs:
            object_key = str(Path(upload_prefix) / output_name)
            file_handle = BytesIO(file_bytes)
            client.storage.from_(bucket).upload(
                object_key,
                file_handle.getvalue(),
                {
                    "content-type": "application/octet-stream",
                    "upsert": "true",
                },
            )
            url = client.storage.from_(bucket).get_public_url(object_key)
            uploaded_files.append({"name": output_name, "path": object_key, "url": url})

        write_status(
            job_id,
            status="complete",
            phase="completed",
            step=total_steps,
            total_steps=total_steps,
            message="Uploaded outputs to Supabase",
            done=True,
            result_type="cloud",
            files=uploaded_files,
        )
        return {"job_id": job_id, "files": uploaded_files, "result_type": "cloud"}

    write_status(
        job_id,
        status="running",
        phase="uploading_or_staging_results",
        step=total_steps,
        total_steps=total_steps,
        message="Staging outputs for download",
        done=False,
    )
    manifest = _save_temp_outputs(
        job_id=job_id,
        outputs=outputs,
        ttl_seconds=ttl_seconds,
        download_base_url=download_base_url,
    )
    files = manifest.get("files", [])
    write_status(
        job_id,
        status="complete",
        phase="completed",
        step=total_steps,
        total_steps=total_steps,
        message="Results ready for download",
        done=True,
        result_type="temp",
        result_files=files,
        results_url=results_url,
        results_path=results_url,
        expires_at=manifest.get("expires_at"),
    )
    return {
        "job_id": job_id,
        "result_type": "temp",
        "files": files,
        "results_url": results_url,
        "expires_at": manifest.get("expires_at"),
    }


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
    _cleanup_expired_results()
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
    if gpu_request not in _GPU_DISPATCH and gpu_request not in {"cpu", "none"}:
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
    )

    ttl_raw = form.get("resultTtlSeconds") or form.get("ttlSeconds")
    try:
        ttl_seconds = int(ttl_raw) if ttl_raw else DEFAULT_RESULT_TTL_SECONDS
    except ValueError:
        raise HTTPException(status_code=400, detail="resultTtlSeconds must be an integer")

    call = _process_job.spawn(
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

    _cleanup_expired_results()
    progress = read_progress(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found")
    return progress


@app.function(
    **_WORKER_KWARGS,
)
@modal.fastapi_endpoint(method="GET")
def get_results(request: Request, job_id: str):
    """Get temporary result manifest for a completed job."""
    unauthorized = _authorize_request(request)
    if unauthorized:
        return unauthorized

    _cleanup_expired_results()
    manifest = _load_manifest(job_id)
    if not manifest:
        raise HTTPException(status_code=404, detail="Results not found")

    progress = read_progress(job_id) or {}
    complete = bool(progress.get("done"))
    return {
        "job_id": job_id,
        "complete": complete,
        "expires_at": manifest.get("expires_at"),
        "files": manifest.get("files", []),
    }


@app.function(
    **_WORKER_KWARGS,
)
@modal.fastapi_endpoint(method="GET")
def download_result(request: Request, job_id: str, name: str, consume: bool = True):
    """Download one temporary result file.

    By default `consume=true`, which deletes the file after it is read.
    """
    import json

    unauthorized = _authorize_request(request)
    if unauthorized:
        return unauthorized

    _cleanup_expired_results()
    manifest = _load_manifest(job_id)
    if not manifest:
        raise HTTPException(status_code=404, detail="Results not found")

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

    file_bytes = file_path.read_bytes()
    media_type = entry.get("content_type") if isinstance(entry, dict) else None
    if not isinstance(media_type, str) or not media_type:
        media_type = mimetypes.guess_type(target_name)[0] or "application/octet-stream"

    if consume:
        try:
            file_path.unlink(missing_ok=True)
            remaining = [
                item
                for item in files
                if isinstance(item, dict) and item.get("name") != target_name
            ]
            manifest["files"] = remaining
            if remaining:
                _manifest_path(job_id).write_text(json.dumps(manifest), encoding="utf-8")
                progress_volume.commit()
            else:
                _remove_job_results(job_id)
        except Exception:
            logger.exception("Failed to consume result file for job_id=%s name=%s", job_id, target_name)

    response = Response(content=file_bytes, media_type=media_type)
    response.headers["Content-Disposition"] = f'attachment; filename="{target_name}"'
    return response

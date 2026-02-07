"""Modal FastAPI endpoint for SHARP inference and Supabase/R2 upload."""

from __future__ import annotations

import logging
import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import Sequence

import modal
from fastapi import HTTPException, Request, Response
from fastapi.responses import StreamingResponse

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
from sharp.modal.progress import PROGRESS_VOLUME_PATH, progress_volume, read_progress, write_progress

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
    # run endpoint on CPU; GPU is selected in worker
    **_WORKER_KWARGS,
)
@modal.fastapi_endpoint(method="POST")
async def process_image(request: Request):
    """Run SHARP on an uploaded image, upload outputs to Supabase/R2, and return URLs."""
    from sharp.modal.app import _predict_batch_impl

    api_key = os.environ.get(API_KEY_ENV)
    if api_key:
        if request.headers.get(API_KEY_HEADER) != api_key:
            return Response(status_code=401)

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
    total_steps = len(image_batch)
    write_progress(
        job_id,
        {
            "status": "queued",
            "step": 0,
            "total_steps": total_steps,
            "message": "Queued",
            "done": False,
        },
    )

    formats_raw = form.get("format") or form.get("formats")
    export_formats: Sequence[str]
    if isinstance(formats_raw, str) and formats_raw.strip():
        export_formats = [fmt.strip() for fmt in formats_raw.split(",") if fmt.strip()]
    else:
        export_formats = DEFAULT_EXPORT_FORMATS

    gpu_request = (form.get("gpu") or form.get("gpu_type") or "a10").strip().lower()
    try:
        if gpu_request in _GPU_DISPATCH:
            outputs = await _GPU_DISPATCH[gpu_request].remote.aio(
                image_batch=image_batch,
                export_formats=export_formats,
                job_id=job_id,
            )
        elif gpu_request in {"cpu", "none"}:
            outputs = _predict_batch_impl(
                image_batch=image_batch,
                export_formats=tuple(export_formats),
                job_id=job_id,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported gpu: {gpu_request}")
    except Exception as e:
        write_progress(
            job_id,
            {
                "status": "failed",
                "step": 0,
                "total_steps": total_steps,
                "message": f"Failed: {e}",
                "done": True,
                "error": str(e),
            },
        )
        raise

    # Explicit storage target: "r2" | "supabase" | anything else -> direct download
    storage_target = (form.get("storageTarget") or "").strip().lower()
    prefix = form.get("prefix") or ""
    logger.info("Received storageTarget=%r, prefix=%r", storage_target, prefix)

    access_string = form.get("accessString") or form.get("access") or ""
    access = _parse_access_string(access_string)

    # -------------------------------------------------------------------------
    # R2 Upload
    # -------------------------------------------------------------------------
    if storage_target == "r2":
        s3_endpoint = access.get("s3Endpoint")
        s3_access_key_id = access.get("s3AccessKeyId")
        s3_secret_access_key = access.get("s3SecretAccessKey")
        s3_bucket = access.get("s3Bucket")
        s3_public_url_base = access.get("s3PublicUrlBase")  # optional custom domain

        missing = []
        if not s3_endpoint:
            missing.append("s3Endpoint")
        if not s3_access_key_id:
            missing.append("s3AccessKeyId")
        if not s3_secret_access_key:
            missing.append("s3SecretAccessKey")
        if not s3_bucket:
            missing.append("s3Bucket")

        if missing:
            logger.warning("R2 selected but missing fields: %s", missing)
            raise HTTPException(
                status_code=400,
                detail=f"storageTarget=r2 requires: {', '.join(missing)}",
            )

        logger.info(
            "Upload target: R2 (s3Endpoint=%s, bucket=%s, prefix=%s)",
            s3_endpoint,
            s3_bucket,
            prefix,
        )
        uploaded_files: list[dict[str, str]] = []
        r2_config = {
            "s3Endpoint": s3_endpoint,
            "s3AccessKeyId": s3_access_key_id,
            "s3SecretAccessKey": s3_secret_access_key,
            "s3Bucket": s3_bucket,
            "prefix": prefix,
            "s3PublicUrlBase": s3_public_url_base,
        }

        for output_name, file_bytes in outputs:
            logger.info("R2 upload start: %s (%d bytes)", output_name, len(file_bytes))
            try:
                url = _upload_to_r2(
                    file_content=file_bytes,
                    filename=output_name,
                    config=r2_config,
                )
                object_key = f"{prefix}/{output_name}".strip("/")
                uploaded_files.append({"name": output_name, "path": object_key, "url": url})
                logger.info("R2 upload complete: %s -> %s", object_key, url)
            except Exception as e:
                logger.exception("R2 upload failed for %s", output_name)
                raise HTTPException(status_code=500, detail=f"R2 upload failed: {e}")

        write_progress(
            job_id,
            {
                "status": "complete",
                "step": total_steps,
                "total_steps": total_steps,
                "message": "Done",
                "done": True,
            },
        )
        return {"job_id": job_id, "files": uploaded_files}

    # -------------------------------------------------------------------------
    # Supabase Upload
    # -------------------------------------------------------------------------
    if storage_target == "supabase":
        supabase_url = access.get("supabaseUrl") or access.get("SUPABASE_URL")
        supabase_key = access.get("supabaseKey") or access.get("SUPABASE_KEY")
        bucket = access.get("supabaseBucket") or access.get("SUPABASE_BUCKET") or DEFAULT_BUCKET

        from supabase import create_client

        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=400, detail="Supabase accessString missing supabaseUrl/supabaseKey.")

        client = create_client(supabase_url, supabase_key)
        prefix = prefix or "collections/default/assets"

        logger.info("Upload target: Supabase (bucket=%s, prefix=%s)", bucket, prefix)
        uploaded_files: list[dict[str, str]] = []
        for output_name, file_bytes in outputs:
            object_key = str(Path(prefix) / output_name)
            file_handle = BytesIO(file_bytes)

            resp = client.storage.from_(bucket).upload(
                object_key,
                file_handle.getvalue(),
                {
                    "content-type": "application/octet-stream",
                    "upsert": "true",  # must be str, not bool
                },
            )
            logger.info("Supabase upload response: %s", resp)

            url = client.storage.from_(bucket).get_public_url(object_key)
            uploaded_files.append({"name": output_name, "path": object_key, "url": url})
            logger.info("Supabase upload complete: %s -> %s", object_key, url)

        write_progress(
            job_id,
            {
                "status": "complete",
                "step": total_steps,
                "total_steps": total_steps,
                "message": "Done",
                "done": True,
            },
        )
        return {"job_id": job_id, "files": uploaded_files}

    # -------------------------------------------------------------------------
    # Default: Direct download (return to sender)
    # -------------------------------------------------------------------------
    logger.info("Return mode: direct download (storageTarget=%r)", storage_target)
    boundary = f"sharp-{uuid.uuid4().hex}"

    def iter_parts():
        for output_name, file_bytes in outputs:
            header = (
                f"--{boundary}\r\n"
                f"Content-Type: application/octet-stream\r\n"
                f"Content-Disposition: attachment; filename=\"{output_name}\"\r\n"
                f"Content-Length: {len(file_bytes)}\r\n\r\n"
            )
            yield header.encode()
            yield file_bytes
            yield b"\r\n"
        yield f"--{boundary}--\r\n".encode()

    write_progress(
        job_id,
        {
            "status": "complete",
            "step": total_steps,
            "total_steps": total_steps,
            "message": "Done",
            "done": True,
        },
    )

    response = StreamingResponse(
        iter_parts(),
        media_type=f"multipart/mixed; boundary={boundary}",
    )
    response.headers["X-Job-Id"] = job_id
    return response


@app.function(
    **_WORKER_KWARGS,
)
@modal.fastapi_endpoint(method="GET")
def get_progress(job_id: str):
    """Return the latest progress snapshot for a job."""
    progress = read_progress(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found")
    return progress

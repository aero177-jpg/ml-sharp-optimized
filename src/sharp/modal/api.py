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
APP_NAME = "ml-sharp-optimizedv2"

# Secret names expected to be provisioned in Modal
SUPABASE_SECRET_NAME = "supabase-creds"
API_AUTH_SECRET_NAME = "sharp-api-auth"

API_KEY_HEADER = "X-API-KEY"
API_KEY_ENV = "API_AUTH_TOKEN"
SUPABASE_URL_ENV = "SUPABASE_URL"
SUPABASE_KEY_ENV = "SUPABASE_KEY"
SUPABASE_BUCKET_ENV = "SUPABASE_BUCKET"

DEFAULT_BUCKET = "testbucket"
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

# Separate Modal app for the HTTP endpoint
app = modal.App(name=APP_NAME)

_WORKER_KWARGS = dict(
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=TIMEOUT_SECONDS,
    image=api_image,
    secrets=[
        modal.Secret.from_name(SUPABASE_SECRET_NAME),
        modal.Secret.from_name(API_AUTH_SECRET_NAME),
    ],
)


def _run_batch_impl(
    image_bytes: bytes, filename: str, export_formats: Sequence[str]
) -> list[tuple[str, bytes]]:
    from sharp.modal.app import _predict_batch_impl

    return _predict_batch_impl(
        image_batch=[(image_bytes, filename)],
        export_formats=tuple(export_formats),
    )


@app.function(gpu="a10", **_WORKER_KWARGS)
def _predict_on_a10(image_bytes: bytes, filename: str, export_formats: Sequence[str]):
    return _run_batch_impl(image_bytes, filename, export_formats)


@app.function(gpu="t4", **_WORKER_KWARGS)
def _predict_on_t4(image_bytes: bytes, filename: str, export_formats: Sequence[str]):
    return _run_batch_impl(image_bytes, filename, export_formats)


@app.function(gpu="l4", **_WORKER_KWARGS)
def _predict_on_l4(image_bytes: bytes, filename: str, export_formats: Sequence[str]):
    return _run_batch_impl(image_bytes, filename, export_formats)


@app.function(gpu="a100", **_WORKER_KWARGS)
def _predict_on_a100(image_bytes: bytes, filename: str, export_formats: Sequence[str]):
    return _run_batch_impl(image_bytes, filename, export_formats)


@app.function(gpu="h100", **_WORKER_KWARGS)
def _predict_on_h100(image_bytes: bytes, filename: str, export_formats: Sequence[str]):
    return _run_batch_impl(image_bytes, filename, export_formats)


_GPU_DISPATCH = {
    "a10": _predict_on_a10,
    "t4": _predict_on_t4,
    "l4": _predict_on_l4,
    "a100": _predict_on_a100,
    "h100": _predict_on_h100,
}


def _upload_to_r2(*, file_content: bytes, filename: str, config: dict[str, str]) -> str:
    import mimetypes

    import boto3
    from botocore.config import Config

    s3 = boto3.client(
        service_name="s3",
        endpoint_url=config["s3Endpoint"],
        aws_access_key_id=config["s3AccessKeyId"],
        aws_secret_access_key=config["s3SecretAccessKey"],
        region_name="auto",
        config=Config(signature_version=R2_SIGNATURE_VERSION),
    )

    prefix = config.get("prefix") or ""
    full_key = f"{prefix}/{filename}".strip("/")
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    resp = s3.put_object(
        Bucket=config["s3Bucket"],
        Key=full_key,
        Body=file_content,
        ContentType=content_type,
    )
    logger.info("R2 put_object response: %s", resp)

    return f"https://{config['s3Bucket']}.r2.cloudflarestorage.com/{full_key}"


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
    upload = form.get("file")
    if upload is None:
        raise HTTPException(status_code=400, detail="Form field 'file' is required.")

    filename = upload.filename or "upload.png"
    image_bytes = await upload.read()

    formats_raw = form.get("format") or form.get("formats")
    export_formats: Sequence[str]
    if isinstance(formats_raw, str) and formats_raw.strip():
        export_formats = [fmt.strip() for fmt in formats_raw.split(",") if fmt.strip()]
    else:
        export_formats = DEFAULT_EXPORT_FORMATS

    gpu_request = (form.get("gpu") or form.get("gpu_type") or "a10").strip().lower()
    if gpu_request in _GPU_DISPATCH:
        outputs = await _GPU_DISPATCH[gpu_request].remote.aio(
            image_bytes=image_bytes,
            filename=filename,
            export_formats=export_formats,
        )
    elif gpu_request in {"cpu", "none"}:
        outputs = _predict_batch_impl(
            image_batch=[(image_bytes, filename)],
            export_formats=tuple(export_formats),
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported gpu: {gpu_request}")

    return_mode = (form.get("return") or "supabase").strip().lower()
    if return_mode in {"direct", "download", "stream"}:
        logger.info("Return mode: direct download/stream (%s)", return_mode)
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

        return StreamingResponse(
            iter_parts(),
            media_type=f"multipart/mixed; boundary={boundary}",
        )

    s3_endpoint = form.get("s3Endpoint")
    s3_access_key_id = form.get("s3AccessKeyId")
    s3_secret_access_key = form.get("s3SecretAccessKey")
    s3_bucket = form.get("s3Bucket")
    prefix = form.get("prefix") or ""

    if s3_endpoint and s3_access_key_id and s3_secret_access_key and s3_bucket:
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
        }

        for output_name, file_bytes in outputs:
            logger.info("R2 upload start: %s (%d bytes)", output_name, len(file_bytes))
            url = _upload_to_r2(
                file_content=file_bytes,
                filename=output_name,
                config=r2_config,
            )
            object_key = f"{prefix}/{output_name}".strip("/")
            uploaded_files.append({"name": output_name, "path": object_key, "url": url})
            logger.info("R2 upload complete: %s -> %s", object_key, url)

        return {"files": uploaded_files}

    supabase_url = os.environ.get(SUPABASE_URL_ENV)
    supabase_key = os.environ.get(SUPABASE_KEY_ENV)
    bucket = os.environ.get(SUPABASE_BUCKET_ENV, DEFAULT_BUCKET)

    from supabase import create_client

    if not supabase_url or not supabase_key:
        raise HTTPException(status_code=500, detail="Supabase credentials not configured.")

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

    return {"files": uploaded_files}

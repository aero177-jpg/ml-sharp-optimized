"""Modal container images for SHARP.

Three images:

- ``create_modal_image``: the CLI image (``sharp cloud predict``). Torch + SHARP deps.
- ``create_worker_image``: the API's GPU worker. CLI deps + storage clients + repo source.
- ``create_web_image``: the API's HTTP layer. No torch, so web containers start fast.

gsplat is not installed anywhere: only ``sharp render`` uses it, and that runs locally.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from pathlib import Path

import modal

PYTHON_VERSION = "3.11"
REMOTE_REPO_PATH = "/root/ml-sharp"
REMOTE_SRC_PATH = f"{REMOTE_REPO_PATH}/src"

_TORCH = ("torch==2.5.1", "torchvision==0.20.1")
_TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
_SHARP_DEPS = (
    "timm>=1.0.0",
    "scipy>=1.11.0",
    "plyfile>=1.0.0",
    "imageio>=2.30.0",
    "pillow>=10.0.0",
    "pillow-heif>=0.16.0",
    "numpy>=1.24.0",
    "click>=8.0.0",
)
_LOCAL_IGNORE = [".git", ".venv", "__pycache__", "node_modules", ".conda", "venv", "output"]


def create_modal_image() -> modal.Image:
    """Image for the CLI's cloud inference functions (``sharp.modal.app``)."""
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .pip_install(*_TORCH, index_url=_TORCH_INDEX)
        .pip_install(*_SHARP_DEPS)
    )


def _with_source(image: modal.Image, repo_root: Path | None, env: dict[str, str]) -> modal.Image:
    image = image.env({"PYTHONPATH": REMOTE_SRC_PATH, "MODAL_IS_REMOTE": "1", **env})
    # add_local_dir must be the last layer. repo_root is None inside the container,
    # where the source is already baked in.
    return image.add_local_dir(
        str(repo_root) if repo_root else "/dummy", REMOTE_REPO_PATH, ignore=_LOCAL_IGNORE
    )


def create_worker_image(repo_root: Path | None, env: dict[str, str] | None = None) -> modal.Image:
    """GPU worker image: torch, SHARP deps, storage clients, repo source."""
    base = (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .pip_install(*_TORCH, index_url=_TORCH_INDEX)
        .pip_install(*_SHARP_DEPS, "fastapi", "supabase", "boto3")
    )
    return _with_source(base, repo_root, env or {})


def create_web_image(repo_root: Path | None, env: dict[str, str] | None = None) -> modal.Image:
    """HTTP image: FastAPI only. Must never import torch at module level."""
    base = modal.Image.debian_slim(python_version=PYTHON_VERSION).pip_install(
        "fastapi[standard]", "python-multipart"
    )
    return _with_source(base, repo_root, env or {})

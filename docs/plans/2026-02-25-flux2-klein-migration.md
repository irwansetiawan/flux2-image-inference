# Flux2.Klein 9B Migration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace WAN 2.2 video generation with Flux2.Klein 9B image generation (txt2img + image editing), synchronous API, S3 output.

**Architecture:** Synchronous FastAPI server (port 8000) submits ComfyUI workflows (port 8188) for Flux2.Klein 9B FP8. No job queue, no database, no background worker. Returns S3 presigned URLs.

**Tech Stack:** FastAPI, ComfyUI, Flux2.Klein 9B FP8, Qwen 3 8B text encoder, boto3/S3

---

## Model Files

| Component | File | Size | HuggingFace Repo | ComfyUI Path |
|-----------|------|------|-------------------|--------------|
| Diffusion (distilled) | `flux-2-klein-9b-fp8.safetensors` | ~9.5GB | `Comfy-Org/flux2-klein-9B` | `models/diffusion_models/` |
| Diffusion (base) | `flux-2-klein-base-9b-fp8.safetensors` | ~9.5GB | `black-forest-labs/FLUX.2-klein-base-9b-fp8` | `models/diffusion_models/` |
| Text Encoder | `qwen_3_8b_fp8mixed.safetensors` | ~8GB | `Comfy-Org/flux2-klein-9B` | `models/text_encoders/` |
| VAE | `flux2-vae.safetensors` | ~0.5GB | `Comfy-Org/flux2-dev` | `models/vae/` |

**VRAM Budget (24GB A10G):** ~9.5 + ~8 + ~0.5 = ~18GB loaded, ~6GB for inference.

## ComfyUI Node Reference

**Text-to-Image (distilled, 4 steps):**
- `UNETLoader` → `CLIPLoader` → `VAELoader`
- `CLIPTextEncode` → `FluxGuidance` (guidance=4.0)
- `ConditioningZeroOut` (negative)
- `EmptyFlux2LatentImage` (width, height)
- `KSampler` (euler, 4 steps, cfg=1.0)
- `VAEDecode` → `SaveImage`

**Image Editing (distilled, 4 steps):**
- Same loaders
- `LoadImage` → `ImageScaleToTotalPixels` → `GetImageSize`
- `VAEEncode` → `ReferenceLatent` (combines image + text conditioning)
- `EmptyFlux2LatentImage` (matching input dimensions)
- `RandomNoise` → `KSamplerSelect` → `Flux2Scheduler`
- `CFGGuider` → `SamplerCustomAdvanced`
- `VAEDecode` → `SaveImage`

---

### Task 1: Delete WAN-specific files and old docs

**Files:**
- Delete: `src/database.py`
- Delete: `src/worker.py`
- Delete: `docs/plans/2026-02-04-wan22-implementation.md`
- Delete: `docs/plans/2026-02-04-wan22-ec2-setup-design.md`

**Step 1: Remove files**

```bash
rm src/database.py src/worker.py
rm docs/plans/2026-02-04-wan22-implementation.md docs/plans/2026-02-04-wan22-ec2-setup-design.md
```

**Step 2: Commit**

```bash
git add -A
git commit -m "chore: remove WAN 2.2 video files (database, worker, old plans)"
```

---

### Task 2: Rewrite config.py for Flux2

**Files:**
- Modify: `src/config.py`

**Step 1: Replace entire config.py**

Replace all WAN/video/MMAudio settings with Flux2 image settings:

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API
API_SECRET_KEY = os.environ["API_SECRET_KEY"]
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# AWS
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
S3_BUCKET = os.environ["S3_BUCKET"]
S3_REGION = os.getenv("S3_REGION", "ap-southeast-1")

# Paths
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
COMFYUI_DIR = DATA_DIR / "ComfyUI"
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Ensure directories exist
INPUTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ComfyUI backend
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")

# Flux2.Klein 9B FP8 model filenames
DIFFUSION_MODEL = "flux-2-klein-9b-fp8.safetensors"
DIFFUSION_MODEL_BASE = "flux-2-klein-base-9b-fp8.safetensors"
TEXT_ENCODER_MODEL = "qwen_3_8b_fp8mixed.safetensors"
VAE_MODEL = "flux2-vae.safetensors"

# Image generation defaults
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 4
DEFAULT_CFG = 1.0
DEFAULT_GUIDANCE = 4.0
DEFAULT_SAMPLER = "euler"
MAX_MEGAPIXELS = 2.0
PRESIGNED_URL_EXPIRY = 3600  # 1 hour
INFERENCE_TIMEOUT = 300  # 5 minutes
```

**Step 2: Commit**

```bash
git add src/config.py
git commit -m "feat: rewrite config for Flux2.Klein 9B FP8"
```

---

### Task 3: Rewrite inference.py for Flux2

**Files:**
- Modify: `src/inference.py`

**Step 1: Replace entire inference.py**

Complete rewrite with two workflow builders (txt2img and image editing):

```python
import json
import shutil
import time
import logging
import urllib.request
import urllib.parse
from pathlib import Path

from src.config import (
    COMFYUI_DIR,
    COMFYUI_URL,
    OUTPUTS_DIR,
    DIFFUSION_MODEL,
    TEXT_ENCODER_MODEL,
    VAE_MODEL,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_STEPS,
    DEFAULT_CFG,
    DEFAULT_GUIDANCE,
    DEFAULT_SAMPLER,
    MAX_MEGAPIXELS,
    INFERENCE_TIMEOUT,
)

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """Raised when image generation fails."""
    pass


def _build_txt2img_workflow(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    guidance: float,
    seed: int,
) -> dict:
    """Build a Flux2.Klein 9B text-to-image workflow for ComfyUI."""
    return {
        # --- Model loading ---
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": DIFFUSION_MODEL,
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER_MODEL,
                "type": "flux",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": VAE_MODEL,
            },
        },
        # --- Text encoding ---
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": prompt,
            },
        },
        # --- Guidance ---
        "5": {
            "class_type": "FluxGuidance",
            "inputs": {
                "conditioning": ["4", 0],
                "guidance": guidance,
            },
        },
        # --- Negative conditioning ---
        "6": {
            "class_type": "ConditioningZeroOut",
            "inputs": {
                "conditioning": ["4", 0],
            },
        },
        # --- Empty latent ---
        "7": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {
                "width": width,
                "height": height,
            },
        },
        # --- Sampling ---
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": DEFAULT_SAMPLER,
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        # --- Decode ---
        "9": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["8", 0],
                "vae": ["3", 0],
            },
        },
        # --- Save ---
        "10": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["9", 0],
                "filename_prefix": "flux2_gen",
            },
        },
    }


def _build_edit_workflow(
    prompt: str,
    negative_prompt: str,
    image_path: Path,
    steps: int,
    cfg: float,
    guidance: float,
    seed: int,
) -> dict:
    """Build a Flux2.Klein 9B image editing workflow for ComfyUI.

    Uses ReferenceLatent to condition generation on the input image.
    """
    return {
        # --- Model loading ---
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": DIFFUSION_MODEL,
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER_MODEL,
                "type": "flux",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": VAE_MODEL,
            },
        },
        # --- Text encoding ---
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": prompt,
            },
        },
        # --- Negative conditioning ---
        "5": {
            "class_type": "ConditioningZeroOut",
            "inputs": {
                "conditioning": ["4", 0],
            },
        },
        # --- Load and scale input image ---
        "6": {
            "class_type": "LoadImage",
            "inputs": {
                "image": image_path.name,
            },
        },
        "7": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["6", 0],
                "upscale_method": "lanczos",
                "megapixels": MAX_MEGAPIXELS,
            },
        },
        "8": {
            "class_type": "GetImageSize",
            "inputs": {
                "image": ["7", 0],
            },
        },
        # --- Encode image to latent ---
        "9": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["7", 0],
                "vae": ["3", 0],
            },
        },
        # --- Reference conditioning ---
        "10": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "reference_latent": ["9", 0],
                "conditioning": ["4", 0],
            },
        },
        # --- Create empty latent matching input size ---
        "11": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {
                "width": ["8", 0],
                "height": ["8", 1],
            },
        },
        # --- Advanced sampling pipeline ---
        "12": {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": seed,
            },
        },
        "13": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": DEFAULT_SAMPLER,
            },
        },
        "14": {
            "class_type": "Flux2Scheduler",
            "inputs": {
                "width": ["8", 0],
                "height": ["8", 1],
                "steps": steps,
            },
        },
        "15": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0],
                "positive": ["10", 0],
                "negative": ["5", 0],
                "cfg": cfg,
            },
        },
        "16": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["12", 0],
                "guider": ["15", 0],
                "sampler": ["13", 0],
                "sigmas": ["14", 0],
                "latent_image": ["11", 0],
            },
        },
        # --- Decode ---
        "17": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["16", 0],
                "vae": ["3", 0],
            },
        },
        # --- Save ---
        "18": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["17", 0],
                "filename_prefix": "flux2_edit",
            },
        },
    }


def _submit_prompt(workflow: dict) -> str:
    """Submit a workflow to ComfyUI and return the prompt_id."""
    payload = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:2000]
        raise InferenceError(f"ComfyUI rejected workflow ({e.code}): {body}")
    except Exception as e:
        raise InferenceError(f"Failed to submit to ComfyUI: {e}")

    if "error" in result:
        raise InferenceError(f"ComfyUI rejected workflow: {result['error']}")

    return result["prompt_id"]


def _poll_completion(prompt_id: str, timeout: int = INFERENCE_TIMEOUT) -> dict:
    """Poll ComfyUI /history until the job completes or times out."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(
                f"{COMFYUI_URL}/history/{prompt_id}", timeout=10
            ) as resp:
                history = json.loads(resp.read())
        except Exception:
            time.sleep(2)
            continue

        if prompt_id in history:
            entry = history[prompt_id]
            if "status" in entry and entry["status"].get("status_str") == "error":
                messages = entry["status"].get("messages", [])
                error_msg = str(messages)[:2000] if messages else "Unknown ComfyUI error"
                raise InferenceError(f"ComfyUI execution failed: {error_msg}")
            return entry

        time.sleep(2)

    raise InferenceError(f"Image generation timed out after {timeout} seconds")


def _download_comfyui_file(item: dict, output_dir: Path) -> Path:
    """Download a single file from ComfyUI output."""
    filename = item["filename"]
    params = urllib.parse.urlencode({
        "filename": filename,
        "subfolder": item.get("subfolder", ""),
        "type": item.get("type", "output"),
    })
    url = f"{COMFYUI_URL}/view?{params}"
    out_path = output_dir / filename
    urllib.request.urlretrieve(url, str(out_path))
    return out_path


def _download_output_image(history_entry: dict, output_dir: Path) -> Path:
    """Download generated image from ComfyUI output."""
    outputs = history_entry.get("outputs", {})

    for node_id, node_output in outputs.items():
        if "images" not in node_output:
            continue
        for item in node_output["images"]:
            if item["filename"].endswith((".png", ".jpg", ".jpeg", ".webp")):
                try:
                    return _download_comfyui_file(item, output_dir)
                except Exception as e:
                    raise InferenceError(f"Failed to download image: {e}")

    raise InferenceError("No image file found in ComfyUI output")


def _copy_image_to_comfyui(image_path: Path) -> None:
    """Copy input image to ComfyUI's input directory for LoadImage node."""
    comfyui_input = COMFYUI_DIR / "input"
    comfyui_input.mkdir(parents=True, exist_ok=True)
    dest = comfyui_input / image_path.name
    shutil.copy2(str(image_path), str(dest))


def generate_image(
    prompt: str,
    negative_prompt: str = "",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    steps: int = DEFAULT_STEPS,
    cfg: float = DEFAULT_CFG,
    guidance: float = DEFAULT_GUIDANCE,
    seed: int | None = None,
) -> tuple[Path, int]:
    """Generate an image from a text prompt.

    Returns (image_path, seed).
    """
    import random
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    output_dir = OUTPUTS_DIR / f"gen_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    workflow = _build_txt2img_workflow(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        guidance=guidance,
        seed=seed,
    )

    logger.info(f"Submitting txt2img to ComfyUI (seed={seed})")
    prompt_id = _submit_prompt(workflow)

    history = _poll_completion(prompt_id)
    image_path = _download_output_image(history, output_dir)
    logger.info(f"Image saved to {image_path}")

    return image_path, seed


def edit_image(
    prompt: str,
    image_path: Path,
    negative_prompt: str = "",
    steps: int = DEFAULT_STEPS,
    cfg: float = DEFAULT_CFG,
    guidance: float = DEFAULT_GUIDANCE,
    seed: int | None = None,
) -> tuple[Path, int]:
    """Edit an image using a text prompt and reference image.

    Returns (image_path, seed).
    """
    import random
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    output_dir = OUTPUTS_DIR / f"edit_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    _copy_image_to_comfyui(image_path)

    workflow = _build_edit_workflow(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_path=image_path,
        steps=steps,
        cfg=cfg,
        guidance=guidance,
        seed=seed,
    )

    logger.info(f"Submitting image edit to ComfyUI (seed={seed})")
    prompt_id = _submit_prompt(workflow)

    history = _poll_completion(prompt_id)
    output_path = _download_output_image(history, output_dir)
    logger.info(f"Edited image saved to {output_path}")

    return output_path, seed


def cleanup_files(image_path: Path, input_path: Path | None = None):
    """Clean up temporary files after generation."""
    if image_path.parent.exists():
        shutil.rmtree(image_path.parent)

    if input_path and input_path.exists():
        comfyui_copy = COMFYUI_DIR / "input" / input_path.name
        if comfyui_copy.exists():
            comfyui_copy.unlink()
        input_path.unlink()
```

**Step 2: Commit**

```bash
git add src/inference.py
git commit -m "feat: rewrite inference for Flux2.Klein 9B (txt2img + editing)"
```

---

### Task 4: Rewrite storage.py for image uploads

**Files:**
- Modify: `src/storage.py`

**Step 1: Replace storage.py**

Change from video uploads to image uploads:

```python
import boto3
from pathlib import Path

from src.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    S3_BUCKET,
    S3_REGION,
    PRESIGNED_URL_EXPIRY,
)


def get_s3_client():
    """Get an S3 client."""
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=S3_REGION,
    )


def upload_image(local_path: Path, s3_key: str) -> str:
    """Upload an image file to S3 and return the S3 key."""
    client = get_s3_client()
    content_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    content_type = content_types.get(local_path.suffix.lower(), "image/png")
    client.upload_file(
        str(local_path),
        S3_BUCKET,
        s3_key,
        ExtraArgs={"ContentType": content_type},
    )
    return s3_key


def get_presigned_url(s3_key: str) -> str:
    """Generate a presigned URL for downloading an image."""
    client = get_s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": s3_key},
        ExpiresIn=PRESIGNED_URL_EXPIRY,
    )


def download_image(url: str, local_path: Path) -> Path:
    """Download an image from URL to local path."""
    import httpx

    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()

    local_path.write_bytes(response.content)
    return local_path
```

**Step 2: Commit**

```bash
git add src/storage.py
git commit -m "feat: update storage for image uploads (remove video support)"
```

---

### Task 5: Rewrite server.py for synchronous image API

**Files:**
- Modify: `src/server.py`

**Step 1: Replace server.py**

Synchronous API with `/generate` (txt2img) and `/edit` (image editing) endpoints:

```python
import base64
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field

from src.config import (
    API_SECRET_KEY,
    INPUTS_DIR,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_STEPS,
    DEFAULT_CFG,
    DEFAULT_GUIDANCE,
)
from src.inference import generate_image, edit_image, cleanup_files, InferenceError
from src.storage import upload_image, get_presigned_url, download_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Flux2.Klein Image Generation API",
    version="2.0.0",
)


# --- Auth ---

def verify_api_key(x_api_key: str = Header(...)):
    """Verify the API key header."""
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# --- Request/Response Models ---

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = Field(default=DEFAULT_WIDTH, ge=256, le=2048)
    height: int = Field(default=DEFAULT_HEIGHT, ge=256, le=2048)
    steps: int = Field(default=DEFAULT_STEPS, ge=1, le=50)
    cfg: float = Field(default=DEFAULT_CFG, ge=0.0, le=20.0)
    guidance: float = Field(default=DEFAULT_GUIDANCE, ge=0.0, le=20.0)
    seed: Optional[int] = None


class EditRequest(BaseModel):
    prompt: str
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    negative_prompt: str = ""
    steps: int = Field(default=DEFAULT_STEPS, ge=1, le=50)
    cfg: float = Field(default=DEFAULT_CFG, ge=0.0, le=20.0)
    guidance: float = Field(default=DEFAULT_GUIDANCE, ge=0.0, le=20.0)
    seed: Optional[int] = None


class ImageResponse(BaseModel):
    image_url: str
    seed: int
    width: int
    height: int
    generation_time_seconds: float


class HealthResponse(BaseModel):
    status: str
    model: str


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", model="flux2-klein-9b-fp8")


@app.post("/generate", response_model=ImageResponse, dependencies=[Depends(verify_api_key)])
async def generate(request: GenerateRequest):
    """Generate an image from a text prompt (synchronous)."""
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    start_time = time.time()
    image_path = None

    try:
        image_path, seed = generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            steps=request.steps,
            cfg=request.cfg,
            guidance=request.guidance,
            seed=request.seed,
        )

        s3_key = f"images/{uuid.uuid4()}{image_path.suffix}"
        upload_image(image_path, s3_key)
        image_url = get_presigned_url(s3_key)

        generation_time = round(time.time() - start_time, 2)

        return ImageResponse(
            image_url=image_url,
            seed=seed,
            width=request.width,
            height=request.height,
            generation_time_seconds=generation_time,
        )

    except InferenceError as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.exception("Unexpected error during generation")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    finally:
        if image_path:
            cleanup_files(image_path)


@app.post("/edit", response_model=ImageResponse, dependencies=[Depends(verify_api_key)])
async def edit(request: EditRequest):
    """Edit an image using a text prompt and reference image (synchronous)."""
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    if not request.image_base64 and not request.image_url:
        raise HTTPException(
            status_code=400,
            detail="Image is required (image_base64 or image_url)",
        )

    start_time = time.time()
    input_path = None
    output_path = None

    try:
        # Handle image input
        input_path = INPUTS_DIR / f"{uuid.uuid4()}.jpg"

        if request.image_base64:
            try:
                image_data = base64.b64decode(request.image_base64)
                input_path.write_bytes(image_data)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")
        else:
            try:
                download_image(request.image_url, input_path)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")

        output_path, seed = edit_image(
            prompt=request.prompt,
            image_path=input_path,
            negative_prompt=request.negative_prompt,
            steps=request.steps,
            cfg=request.cfg,
            guidance=request.guidance,
            seed=request.seed,
        )

        s3_key = f"images/{uuid.uuid4()}{output_path.suffix}"
        upload_image(output_path, s3_key)
        image_url = get_presigned_url(s3_key)

        generation_time = round(time.time() - start_time, 2)

        # Get output dimensions from the image
        from PIL import Image
        with Image.open(output_path) as img:
            width, height = img.size

        return ImageResponse(
            image_url=image_url,
            seed=seed,
            width=width,
            height=height,
            generation_time_seconds=generation_time,
        )

    except HTTPException:
        raise

    except InferenceError as e:
        logger.error(f"Edit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.exception("Unexpected error during edit")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    finally:
        if output_path:
            cleanup_files(output_path, input_path)
```

**Step 2: Commit**

```bash
git add src/server.py
git commit -m "feat: rewrite server for synchronous Flux2 image API"
```

---

### Task 6: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Add Pillow dependency (for reading output image dimensions)**

```
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-dotenv==1.0.0
boto3==1.34.0
httpx==0.26.0
pydantic==2.5.3
Pillow==11.1.0
```

**Step 2: Commit**

```bash
git add requirements.txt
git commit -m "feat: add Pillow dependency for image dimension reading"
```

---

### Task 7: Rewrite setup.sh for Flux2 models

**Files:**
- Modify: `scripts/setup.sh`

**Step 1: Replace setup.sh**

New setup script that downloads Flux2 models and installs no custom nodes (all nodes are native ComfyUI):

```bash
#!/bin/bash
set -e

echo "=== Flux2.Klein 9B ComfyUI Setup ==="

# Configuration
DATA_DIR="${DATA_DIR:-/data}"
COMFYUI_DIR="$DATA_DIR/ComfyUI"

# Create data directories
echo "Creating directories..."
sudo mkdir -p "$DATA_DIR"
sudo chown -R ubuntu:ubuntu "$DATA_DIR"
mkdir -p "$DATA_DIR/inputs" "$DATA_DIR/outputs"

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# =============================================
# ComfyUI Installation
# =============================================
echo "Installing ComfyUI..."
if [ ! -d "$COMFYUI_DIR" ]; then
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
fi

cd "$COMFYUI_DIR"
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
echo "Installing PyTorch..."
pip install --upgrade pip
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124

# Install ComfyUI dependencies
echo "Installing ComfyUI dependencies..."
pip install -r requirements.txt

# =============================================
# Download Flux2.Klein 9B FP8 Models (~18.5GB total)
# =============================================
echo "Downloading Flux2.Klein 9B FP8 models (~18.5GB total)..."
pip install "huggingface_hub[cli]"

MODELS_DIR="$COMFYUI_DIR/models"
HF_DOWNLOAD="$COMFYUI_DIR/venv/bin/python -c"

# Download helper: downloads from HuggingFace and symlinks into ComfyUI model dirs
download_model() {
    local repo="$1"
    local remote_path="$2"
    local target_dir="$3"
    local filename=$(basename "$remote_path")
    local target="$MODELS_DIR/$target_dir/$filename"

    if [ -f "$target" ] || [ -L "$target" ]; then
        echo "  SKIP $filename (already exists)"
        return
    fi

    echo "  Downloading $filename..."
    local local_path=$($HF_DOWNLOAD "
from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id='$repo', filename='$remote_path'))
")
    mkdir -p "$MODELS_DIR/$target_dir"
    ln -sf "$local_path" "$target"
    echo "  -> $target"
}

# Diffusion model - distilled (~9.5GB FP8)
echo "Downloading Flux2.Klein 9B distilled (FP8)..."
download_model "Comfy-Org/flux2-klein-9B" \
    "flux-2-klein-9b-fp8.safetensors" "diffusion_models"

# Diffusion model - base (~9.5GB FP8)
echo "Downloading Flux2.Klein base 9B (FP8)..."
download_model "black-forest-labs/FLUX.2-klein-base-9b-fp8" \
    "flux-2-klein-base-9b-fp8.safetensors" "diffusion_models"

# Text encoder - Qwen 3 8B FP8 mixed (~8GB)
echo "Downloading Qwen 3 8B text encoder..."
download_model "Comfy-Org/flux2-klein-9B" \
    "qwen_3_8b_fp8mixed.safetensors" "text_encoders"

# VAE (~0.5GB)
echo "Downloading Flux2 VAE..."
download_model "Comfy-Org/flux2-dev" \
    "flux2-vae.safetensors" "vae"

# =============================================
# API Server Dependencies
# =============================================
echo "Installing API dependencies..."
cd /home/ubuntu/flux2
python3 -m venv /data/venv
source /data/venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# =============================================
# Add swap space (16GB RAM on g5.xlarge is tight)
# =============================================
if [ ! -f /swapfile ]; then
    echo "Adding 16GB swap..."
    sudo fallocate -l 16G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# =============================================
# Systemd Services
# =============================================
echo "Setting up systemd services..."

# ComfyUI service
sudo tee /etc/systemd/system/comfyui.service > /dev/null <<EOF
[Unit]
Description=ComfyUI Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$COMFYUI_DIR
ExecStart=$COMFYUI_DIR/venv/bin/python main.py --listen 127.0.0.1 --port 8188
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Flux2 API service
sudo tee /etc/systemd/system/flux2-api.service > /dev/null <<EOF
[Unit]
Description=Flux2.Klein Image Generation API
After=network.target comfyui.service
Requires=comfyui.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/flux2
Environment=PATH=/data/venv/bin:/usr/local/bin:/usr/bin:/bin
Environment=DATA_DIR=$DATA_DIR
EnvironmentFile=/home/ubuntu/flux2/.env
ExecStart=/data/venv/bin/uvicorn src.server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable comfyui flux2-api

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Copy your project to /home/ubuntu/flux2"
echo "2. Create .env file with required variables"
echo "3. Start services:"
echo "   sudo systemctl start comfyui"
echo "   sudo systemctl start flux2-api"
echo "4. Check logs:"
echo "   sudo journalctl -u comfyui -f"
echo "   sudo journalctl -u flux2-api -f"
```

**Step 2: Commit**

```bash
git add scripts/setup.sh
git commit -m "feat: rewrite setup for Flux2.Klein 9B FP8 models"
```

---

### Task 8: Update start.sh and stop.sh

**Files:**
- Modify: `scripts/start.sh`
- Modify: `scripts/stop.sh`

**Step 1: Update start.sh**

```bash
#!/bin/bash
set -e

echo "Starting Flux2 services..."

# Start ComfyUI backend first
sudo systemctl start comfyui
echo "ComfyUI started (port 8188)"

# Wait briefly for ComfyUI to initialize
sleep 5

# Start API server
sudo systemctl start flux2-api
echo "API server started (port 8000)"

echo ""
echo "Services running. Check logs with:"
echo "  sudo journalctl -u comfyui -f"
echo "  sudo journalctl -u flux2-api -f"
```

**Step 2: Update stop.sh**

```bash
#!/bin/bash

echo "Stopping Flux2 services..."

sudo systemctl stop flux2-api 2>/dev/null || true
echo "API server stopped"

sudo systemctl stop comfyui 2>/dev/null || true
echo "ComfyUI stopped"

echo "All services stopped"
```

**Step 3: Commit**

```bash
git add scripts/start.sh scripts/stop.sh
git commit -m "feat: update start/stop scripts for Flux2"
```

---

### Task 9: Update README.md

**Files:**
- Modify: `README.md`

**Step 1: Replace README.md**

```markdown
# Flux2.Klein Image Generation API

Self-hosted Flux2.Klein 9B FP8 image generation API powered by ComfyUI on AWS EC2. Supports text-to-image generation and image editing.

## Architecture

- **Model**: Flux2.Klein 9B FP8 (distilled 4-step, ~9.5GB)
- **Text Encoder**: Qwen 3 8B FP8 mixed (~8GB)
- **Backend**: ComfyUI (localhost:8188) — handles model loading, inference, GPU management
- **API**: FastAPI (port 8000) — synchronous REST API
- **EC2 (Compute)**: g5.xlarge — NVIDIA A10G 24GB VRAM
- **S3 (Storage)**: S3 bucket for generated images

## Prerequisites

- AWS EC2 g5.xlarge instance (24GB VRAM A10G GPU)
- Ubuntu 22.04 Deep Learning AMI
- Storage: 100GB EBS (models ~18.5GB + working space)
- S3 bucket for image storage

## Quick Start

### 1. Launch EC2 Instance

Launch a g5.xlarge instance with:
- AMI: Deep Learning AMI GPU PyTorch 2.0+ (Ubuntu 22.04)
- Storage: 100GB EBS
- Security Group: Allow ports 22 (SSH) and 8000 (API)

### 2. Clone and Setup

```bash
ssh -i your-key.pem ubuntu@<ec2-ip>
git clone <your-repo-url> ~/flux2
cd ~/flux2
./scripts/setup.sh
```

### 3. Configure Environment

Create `.env` file:

```bash
cat > .env << 'EOF'
API_SECRET_KEY=your-secret-key-here
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
S3_BUCKET=your-bucket-name
S3_REGION=your-s3-region
EOF
```

### 4. Start Services

```bash
sudo systemctl start comfyui
sudo systemctl start flux2-api
```

## API Usage

See [docs/API.md](docs/API.md) for full API reference.

### Quick Examples

```bash
# Health check
curl http://<ec2-ip>:8000/health

# Text-to-Image
curl -X POST http://<ec2-ip>:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "prompt": "A serene mountain landscape at golden hour",
    "width": 1024,
    "height": 1024
  }'

# Image Editing
curl -X POST http://<ec2-ip>:8000/edit \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "prompt": "Change the sky to a starry night",
    "image_url": "https://example.com/photo.jpg"
  }'
```

## Image Output

- **Resolution**: Up to 2 megapixels (default 1024x1024)
- **Format**: PNG
- **Generation time**: ~12-30s depending on resolution and steps
- **Storage**: Images uploaded to S3, served via presigned URLs (1-hour expiry)

## Troubleshooting

```bash
# Check logs
sudo journalctl -u comfyui -f
sudo journalctl -u flux2-api -f

# Restart services
sudo systemctl restart comfyui
sleep 10
sudo systemctl restart flux2-api

# Check GPU usage
nvidia-smi
```
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for Flux2.Klein image API"
```

---

### Task 10: Update docs/API.md

**Files:**
- Modify: `docs/API.md`

**Step 1: Replace docs/API.md**

```markdown
# Flux2.Klein Image Generation API

REST API for generating and editing images using Flux2.Klein 9B FP8 (powered by ComfyUI).

## Base URL

```
http://<your-server-ip>:8000
```

## Authentication

All endpoints (except `/health`) require an API key via the `X-API-Key` header.

```
X-API-Key: <your-api-key>
```

## Endpoints

### Health Check

```
GET /health
```

**Response**

```json
{
  "status": "ok",
  "model": "flux2-klein-9b-fp8"
}
```

---

### Generate Image (Text-to-Image)

Generate an image from a text prompt. Synchronous — blocks until the image is ready.

```
POST /generate
```

**Request Body**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | Yes | — | Text description of the image to generate |
| `negative_prompt` | string | No | `""` | What to avoid in the image |
| `width` | integer | No | 1024 | Image width (256–2048) |
| `height` | integer | No | 1024 | Image height (256–2048) |
| `steps` | integer | No | 4 | Denoising steps (1–50) |
| `cfg` | float | No | 1.0 | Classifier-free guidance scale (0–20) |
| `guidance` | float | No | 4.0 | Flux guidance scale (0–20) |
| `seed` | integer | No | random | Seed for reproducibility |

**Example**

```bash
curl -X POST http://<ip>:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <key>" \
  -d '{
    "prompt": "A serene mountain landscape at golden hour",
    "width": 1024,
    "height": 1024
  }'
```

**Response**

```json
{
  "image_url": "https://s3.amazonaws.com/...",
  "seed": 827461953,
  "width": 1024,
  "height": 1024,
  "generation_time_seconds": 12.34
}
```

---

### Edit Image

Edit an existing image using a text prompt. Synchronous.

```
POST /edit
```

**Request Body**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | Yes | — | Text description of the desired edit |
| `image_url` | string | Yes* | — | URL of the input image |
| `image_base64` | string | Yes* | — | Base64-encoded input image |
| `negative_prompt` | string | No | `""` | What to avoid |
| `steps` | integer | No | 4 | Denoising steps (1–50) |
| `cfg` | float | No | 1.0 | CFG scale (0–20) |
| `guidance` | float | No | 4.0 | Flux guidance scale (0–20) |
| `seed` | integer | No | random | Seed for reproducibility |

> *Either `image_url` or `image_base64` must be provided.

**Example**

```bash
curl -X POST http://<ip>:8000/edit \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <key>" \
  -d '{
    "prompt": "Change the sky to a starry night",
    "image_url": "https://example.com/photo.jpg"
  }'
```

**Response** (same format as `/generate`)

```json
{
  "image_url": "https://s3.amazonaws.com/...",
  "seed": 12345678,
  "width": 1024,
  "height": 768,
  "generation_time_seconds": 15.67
}
```

---

## Error Responses

| Status | Description |
|--------|-------------|
| 400 | Missing prompt, missing image, invalid base64, failed download |
| 401 | Invalid or missing API key |
| 500 | Inference error or unexpected server error |

## Image Output

- **Model:** Flux2.Klein 9B FP8 Distilled (4-step denoising)
- **Format:** PNG
- **Max Resolution:** 2 megapixels
- **Storage:** S3 with presigned URLs (1-hour expiry)
- **Generation Time:** ~12-30 seconds

## Client Example (Python)

```python
import requests

API_URL = "http://<your-server-ip>:8000"
HEADERS = {"X-API-Key": "<your-api-key>"}

# Text-to-Image
res = requests.post(
    f"{API_URL}/generate",
    headers=HEADERS,
    json={
        "prompt": "A serene mountain landscape at golden hour",
        "width": 1024,
        "height": 1024,
    },
)
print("Image URL:", res.json()["image_url"])

# Image Editing
res = requests.post(
    f"{API_URL}/edit",
    headers=HEADERS,
    json={
        "prompt": "Change the sky to a starry night",
        "image_url": "https://example.com/photo.jpg",
    },
)
print("Edited image:", res.json()["image_url"])
```
```

**Step 2: Commit**

```bash
git add docs/API.md
git commit -m "docs: rewrite API docs for Flux2.Klein image API"
```

---

### Task 11: Update .env and S3 bucket naming

**Files:**
- Modify: `.env` (user must update manually)

**Step 1: Note for user**

The `.env` file should be updated to use a new S3 bucket name. The `S3_BUCKET` value `wan22-bucket-sg` should be changed to something appropriate like `flux2-bucket-sg`. This is a manual step — do not commit `.env`.

**Step 2: No commit needed** (`.env` is gitignored)

---

### Task 12: Clean up __init__.py imports

**Files:**
- Modify: `src/__init__.py`

**Step 1: Verify __init__.py is empty or has no stale imports**

The current `src/__init__.py` should be empty (package marker only). Verify and leave as-is.

**Step 2: No commit needed if unchanged**

---

## Summary of Changes

| File | Action | Description |
|------|--------|-------------|
| `src/config.py` | Rewrite | Flux2 model paths, image defaults |
| `src/inference.py` | Rewrite | txt2img + editing ComfyUI workflows |
| `src/storage.py` | Rewrite | Image uploads (was video) |
| `src/server.py` | Rewrite | Synchronous API, two endpoints |
| `src/database.py` | Delete | No job queue needed |
| `src/worker.py` | Delete | No background processing |
| `scripts/setup.sh` | Rewrite | Flux2 models, no custom nodes |
| `scripts/start.sh` | Update | New service name |
| `scripts/stop.sh` | Update | New service name |
| `requirements.txt` | Update | Add Pillow |
| `README.md` | Rewrite | Flux2 documentation |
| `docs/API.md` | Rewrite | New API reference |
| `docs/plans/2026-02-04-*.md` | Delete | Old WAN plans |

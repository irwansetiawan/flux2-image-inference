import json
import random
import shutil
import time
import logging
import urllib.request
import urllib.parse
import uuid
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
    width: int,
    height: int,
    steps: int,
    cfg: float,
    guidance: float,
    seed: int,
) -> dict:
    """Build a Flux2.Klein 9B text-to-image workflow for ComfyUI."""
    return {
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
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": prompt,
            },
        },
        "5": {
            "class_type": "FluxGuidance",
            "inputs": {
                "conditioning": ["4", 0],
                "guidance": guidance,
            },
        },
        "6": {
            "class_type": "ConditioningZeroOut",
            "inputs": {
                "conditioning": ["4", 0],
            },
        },
        "7": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {
                "width": width,
                "height": height,
            },
        },
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
        "9": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["8", 0],
                "vae": ["3", 0],
            },
        },
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
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": prompt,
            },
        },
        "5": {
            "class_type": "ConditioningZeroOut",
            "inputs": {
                "conditioning": ["4", 0],
            },
        },
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
        "9": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["7", 0],
                "vae": ["3", 0],
            },
        },
        "10": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "reference_latent": ["9", 0],
                "conditioning": ["4", 0],
            },
        },
        "11": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {
                "width": ["8", 0],
                "height": ["8", 1],
            },
        },
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
        "17": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["16", 0],
                "vae": ["3", 0],
            },
        },
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
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    output_dir = OUTPUTS_DIR / f"gen_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    workflow = _build_txt2img_workflow(
        prompt=prompt,
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
    steps: int = DEFAULT_STEPS,
    cfg: float = DEFAULT_CFG,
    guidance: float = DEFAULT_GUIDANCE,
    seed: int | None = None,
) -> tuple[Path, int]:
    """Edit an image using a text prompt and reference image.

    Returns (image_path, seed).
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    output_dir = OUTPUTS_DIR / f"edit_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    _copy_image_to_comfyui(image_path)

    workflow = _build_edit_workflow(
        prompt=prompt,
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

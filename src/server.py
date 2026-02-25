import base64
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from PIL import Image
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
def generate(request: GenerateRequest):
    """Generate an image from a text prompt (synchronous)."""
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    start_time = time.time()
    image_path = None

    try:
        image_path, seed = generate_image(
            prompt=request.prompt,
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
def edit(request: EditRequest):
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
            steps=request.steps,
            cfg=request.cfg,
            guidance=request.guidance,
            seed=request.seed,
        )

        s3_key = f"images/{uuid.uuid4()}{output_path.suffix}"
        upload_image(output_path, s3_key)
        image_url = get_presigned_url(s3_key)

        generation_time = round(time.time() - start_time, 2)

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

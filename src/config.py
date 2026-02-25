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

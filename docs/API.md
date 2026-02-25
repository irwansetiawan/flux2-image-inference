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

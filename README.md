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

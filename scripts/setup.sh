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
download_model "black-forest-labs/FLUX.2-klein-9b-fp8" \
    "flux-2-klein-9b-fp8.safetensors" "diffusion_models"

# Diffusion model - base (~9.5GB FP8)
echo "Downloading Flux2.Klein base 9B (FP8)..."
download_model "black-forest-labs/FLUX.2-klein-base-9b-fp8" \
    "flux-2-klein-base-9b-fp8.safetensors" "diffusion_models"

# Text encoder - Qwen 3 8B FP8 mixed (~8GB)
echo "Downloading Qwen 3 8B text encoder..."
download_model "Comfy-Org/flux2-klein-9B" \
    "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors" "text_encoders"

# VAE (~0.5GB)
echo "Downloading Flux2 VAE..."
download_model "Comfy-Org/flux2-klein-9B" \
    "split_files/vae/flux2-vae.safetensors" "vae"

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

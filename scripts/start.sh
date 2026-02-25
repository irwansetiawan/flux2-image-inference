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

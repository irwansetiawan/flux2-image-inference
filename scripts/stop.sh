#!/bin/bash

echo "Stopping Flux2 services..."

sudo systemctl stop flux2-api 2>/dev/null || true
echo "API server stopped"

sudo systemctl stop comfyui 2>/dev/null || true
echo "ComfyUI stopped"

echo "All services stopped"

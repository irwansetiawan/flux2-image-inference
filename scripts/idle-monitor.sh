#!/bin/bash
# Check if GPU instance has been idle and shut down if so.
# Runs via cron every 5 minutes.

TIMESTAMP_FILE="/tmp/flux2-last-request"
IDLE_MINUTES=30

# If file doesn't exist, create it (instance just booted, give grace period)
if [ ! -f "$TIMESTAMP_FILE" ]; then
    touch "$TIMESTAMP_FILE"
    exit 0
fi

FILE_AGE=$(( $(date +%s) - $(stat -c %Y "$TIMESTAMP_FILE") ))
IDLE_THRESHOLD=$(( IDLE_MINUTES * 60 ))

if [ "$FILE_AGE" -gt "$IDLE_THRESHOLD" ]; then
    logger -t flux2-idle "Instance idle for ${IDLE_MINUTES}+ minutes. Shutting down."
    curl -s -X POST https://n8n.irwansetiawan.com/webhook/ba4845ed-9870-4d0c-b9ad-acd24d31b021 --max-time 10 || true
    sudo shutdown -h now
fi

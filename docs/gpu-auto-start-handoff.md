# GPU Auto-Start Handoff

The GPU instance (g5.xlarge) automatically shuts itself down after 30 minutes of idle time. A separate service needs to start it on demand when inference requests arrive.

## Instance Details

| Setting | Value |
|---------|-------|
| Instance ID | `i-XXXXXXXXXXXXXXXXX` (update after deployment) |
| Region | `us-east-1` (update if different) |
| Instance type | `g5.xlarge` |
| Health endpoint | `http://<private-ip>:8000/health` |
| API port | `8000` |

## IAM Permissions Required

The calling service's IAM role needs:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ec2:StartInstances",
                "ec2:StopInstances",
                "ec2:DescribeInstances"
            ],
            "Resource": "arn:aws:ec2:REGION:ACCOUNT_ID:instance/INSTANCE_ID"
        },
        {
            "Effect": "Allow",
            "Action": "ec2:DescribeInstances",
            "Resource": "*"
        }
    ]
}
```

## How It Works

### Auto-Stop (this project)
- Every `/generate` and `/edit` request touches `/tmp/flux2-last-request`
- A cron job runs every 5 minutes checking that file's age
- If idle > 30 minutes, the instance shuts itself down via `sudo shutdown -h now`
- On boot, the file is touched so the 30-min window starts from boot time

### Auto-Start (caller side)
The calling service should:
1. Check instance state before forwarding inference requests
2. If stopped, start the instance and wait for it to become healthy
3. Forward the request once `/health` returns `200`

## Example Python Code

```python
import time
import boto3
import requests

INSTANCE_ID = "i-XXXXXXXXXXXXXXXXX"
REGION = "us-east-1"
HEALTH_URL = "http://{private_ip}:8000/health"
HEALTH_TIMEOUT = 300  # seconds to wait for healthy state
HEALTH_INTERVAL = 10  # seconds between health checks

ec2 = boto3.client("ec2", region_name=REGION)


def get_instance_state() -> str:
    """Get current instance state: pending, running, stopping, stopped, etc."""
    resp = ec2.describe_instances(InstanceIds=[INSTANCE_ID])
    return resp["Reservations"][0]["Instances"][0]["State"]["Name"]


def get_private_ip() -> str:
    """Get the instance's private IP address."""
    resp = ec2.describe_instances(InstanceIds=[INSTANCE_ID])
    return resp["Reservations"][0]["Instances"][0]["PrivateIpAddress"]


def start_instance() -> str:
    """Start the instance and return its private IP once running."""
    state = get_instance_state()

    if state == "running":
        return get_private_ip()

    if state == "stopped":
        ec2.start_instances(InstanceIds=[INSTANCE_ID])
    elif state == "stopping":
        # Wait for it to fully stop, then start
        ec2.get_waiter("instance_stopped").wait(InstanceIds=[INSTANCE_ID])
        ec2.start_instances(InstanceIds=[INSTANCE_ID])

    ec2.get_waiter("instance_running").wait(InstanceIds=[INSTANCE_ID])
    return get_private_ip()


def wait_for_healthy(private_ip: str) -> bool:
    """Poll /health until the service is ready."""
    url = HEALTH_URL.format(private_ip=private_ip)
    deadline = time.time() + HEALTH_TIMEOUT

    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(HEALTH_INTERVAL)

    return False


def ensure_gpu_ready() -> str:
    """Start instance if needed and wait for healthy state. Returns base URL."""
    private_ip = start_instance()
    if not wait_for_healthy(private_ip):
        raise RuntimeError(f"GPU instance {INSTANCE_ID} did not become healthy")
    return f"http://{private_ip}:8000"
```

## Usage in Calling Service

```python
# Before forwarding an inference request:
base_url = ensure_gpu_ready()
response = requests.post(
    f"{base_url}/generate",
    json={"prompt": "a photo of a cat"},
    headers={"X-API-Key": API_KEY},
    timeout=600,
)
```

## Notes

- The GPU instance takes ~2-3 minutes to boot and load models (cold start)
- Health checks should use a generous timeout (5+ minutes)
- The instance uses a private IP within the VPC — the calling service must be in the same VPC or have connectivity via peering/VPN
- First inference after boot may take longer due to model loading into GPU memory

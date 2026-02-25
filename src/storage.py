import boto3
from pathlib import Path

from src.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    S3_BUCKET,
    S3_REGION,
    PRESIGNED_URL_EXPIRY,
)


def get_s3_client():
    """Get an S3 client."""
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=S3_REGION,
    )


def upload_image(local_path: Path, s3_key: str) -> str:
    """Upload an image file to S3 and return the S3 key."""
    client = get_s3_client()
    content_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    content_type = content_types.get(local_path.suffix.lower(), "image/png")
    client.upload_file(
        str(local_path),
        S3_BUCKET,
        s3_key,
        ExtraArgs={"ContentType": content_type},
    )
    return s3_key


def get_presigned_url(s3_key: str) -> str:
    """Generate a presigned URL for downloading an image."""
    client = get_s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": s3_key},
        ExpiresIn=PRESIGNED_URL_EXPIRY,
    )


def download_image(url: str, local_path: Path) -> Path:
    """Download an image from URL to local path."""
    import httpx

    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()

    local_path.write_bytes(response.content)
    return local_path

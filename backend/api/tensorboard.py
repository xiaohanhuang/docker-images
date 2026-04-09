"""TensorBoard API — proxy and log discovery."""

import asyncio
import logging
import os

from fastapi import APIRouter, HTTPException

from .svc_proxy import SERVICES

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tensorboard"])


def _list_s3_runs(bucket: str) -> list[dict[str, str]]:
    """Paginate through all TensorBoard run prefixes in S3 (sync)."""
    import boto3

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    runs: list[dict[str, str]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix="tensorboard/", Delimiter="/"):
        for p in page.get("CommonPrefixes", []):
            runs.append(
                {
                    "execution_id": p["Prefix"].rstrip("/").split("/")[-1],
                    "s3_path": f"s3://{bucket}/{p['Prefix']}",
                }
            )
    return runs


@router.get("/runs")
async def list_runs():
    """List available TensorBoard log directories in S3."""
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        raise HTTPException(status_code=500, detail="S3_BUCKET not configured")

    runs = await asyncio.to_thread(_list_s3_runs, bucket)
    return {"runs": runs}


@router.get("/url")
async def get_tensorboard_url():
    """Return the browser-reachable TensorBoard URL.

    Always returns the ingress (external) URL — the cluster-internal DNS
    is not accessible from the user's browser.
    """
    try:
        svc = SERVICES["tensorboard"]
    except KeyError:
        raise HTTPException(
            status_code=503,
            detail="TensorBoard service not registered",
        )
    url = svc.ingress_url
    if not url:
        logger.warning("TensorBoard ingress URL is not configured")
        raise HTTPException(
            status_code=503,
            detail="TensorBoard ingress URL not configured",
        )
    return {"url": url}

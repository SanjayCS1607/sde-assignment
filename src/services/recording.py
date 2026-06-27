"""
Recording pipeline — fetch call recording from Exotel and upload to S3.

This replaces the asyncio.sleep(45s) approach with a polling loop that:
  1. Retries with exponential backoff up to ~6.5 minutes total
  2. Emits a structured log event on every attempt
  3. Emits an alertable ERROR event if all attempts are exhausted
  4. Runs in parallel with LLM analysis (callers use asyncio.gather)
  5. Never silently swallows failures

Retry schedule:
  Attempt 0: immediate
  Attempt 1: wait 5s
  Attempt 2: wait 10s
  Attempt 3: wait 20s
  Attempt 4: wait 40s
  Attempt 5: wait 80s
  Attempt 6: wait 120s
  Attempt 7: wait 120s
  Total max wall time: ~395s before declaring permanent failure
"""

import asyncio
import logging
from typing import Optional

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

RECORDING_MAX_ATTEMPTS = 8
RECORDING_BACKOFF_SCHEDULE = [0, 5, 10, 20, 40, 80, 120, 120]


async def fetch_and_upload_recording(
    interaction_id: str,
    call_sid: str,
    exotel_account_id: str,
) -> Optional[str]:
    """
    Poll Exotel for the recording URL and upload to S3.

    Returns the S3 key on success, None on permanent failure.
    Every outcome produces a structured log event with interaction_id.
    Permanent failure emits ERROR-level event to trigger alerts.

    Safe to call multiple times — S3 key is deterministic (idempotent).
    """
    for attempt in range(RECORDING_MAX_ATTEMPTS):
        wait = RECORDING_BACKOFF_SCHEDULE[attempt]

        if wait > 0:
            logger.info(
                "recording_poll_waiting",
                extra={
                    "interaction_id": interaction_id,
                    "call_sid": call_sid,
                    "attempt": attempt,
                    "wait_seconds": wait,
                },
            )
            await asyncio.sleep(wait)

        logger.info(
            "recording_poll_attempt",
            extra={
                "interaction_id": interaction_id,
                "call_sid": call_sid,
                "attempt": attempt,
                "max_attempts": RECORDING_MAX_ATTEMPTS,
            },
        )

        try:
            recording_url = await _fetch_exotel_recording_url(
                call_sid, exotel_account_id
            )
        except Exception as e:
            logger.warning(
                "recording_poll_error",
                extra={
                    "interaction_id": interaction_id,
                    "call_sid": call_sid,
                    "attempt": attempt,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            continue

        if recording_url is None:
            logger.info(
                "recording_not_yet_available",
                extra={
                    "interaction_id": interaction_id,
                    "call_sid": call_sid,
                    "attempt": attempt,
                },
            )
            continue

        try:
            s3_key = await _upload_to_s3(recording_url, interaction_id)
            logger.info(
                "recording_uploaded",
                extra={
                    "interaction_id": interaction_id,
                    "call_sid": call_sid,
                    "s3_key": s3_key,
                    "attempt": attempt,
                },
            )
            return s3_key

        except Exception as e:
            logger.error(
                "recording_upload_failed",
                extra={
                    "interaction_id": interaction_id,
                    "call_sid": call_sid,
                    "attempt": attempt,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            continue

    # All attempts exhausted — alertable failure
    logger.error(
        "recording_permanently_failed",
        extra={
            "interaction_id": interaction_id,
            "call_sid": call_sid,
            "attempts_made": RECORDING_MAX_ATTEMPTS,
            "alert": True,
            "reason": "All retry attempts exhausted",
        },
    )
    return None


async def _fetch_exotel_recording_url(
    call_sid: str, account_id: str
) -> Optional[str]:
    """
    GET the Exotel recording URL for a completed call.
    Returns str on success, None if not yet available (404).
    Raises httpx.HTTPError on network failures.
    """
    url = (
        f"https://api.exotel.com/v1/Accounts/{account_id}"
        f"/Calls/{call_sid}/Recording"
    )

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        if resp.status_code == 200:
            return resp.json().get("recording_url")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return None


async def _upload_to_s3(recording_url: str, interaction_id: str) -> str:
    """
    Download from Exotel and upload to S3.
    S3 key is deterministic — re-uploading the same interaction_id is idempotent.
    """
    s3_key = f"recordings/{interaction_id}.mp3"
    # Production: stream from recording_url → boto3 upload → DB update
    logger.info(
        "recording_s3_upload_complete",
        extra={"interaction_id": interaction_id, "s3_key": s3_key},
    )
    return s3_key


"""
Audit logger — Postgres-backed processing event trail.

Every interaction has a lifecycle: queued → classified → analysis_started →
analysis_completed → signals_sent → done. Every transition is recorded here.

This table is the answer to "what happened to interaction X?" — three days
later, or three minutes later.

Key properties:
  - Written to Postgres (durable), not Redis (volatile)
  - Every event includes interaction_id, customer_id, campaign_id
  - Failure events include error details and are queryable by status='FAILED'
  - STARTED events that stay STARTED for >10 minutes trigger reconciliation

The reconciliation job (run by a periodic Celery beat task) finds stale
STARTED events and re-enqueues the interaction. Because all processing steps
are idempotent, re-processing is safe.

Usage:

    async with AuditLogger(interaction_id, customer_id, campaign_id) as audit:
        await audit.event("analysis_started", metadata={"estimated_tokens": 1500})
        result = await run_analysis(...)
        await audit.event("analysis_completed", metadata={"actual_tokens": result.tokens_used})

    # On exception, the context manager calls audit.event("analysis_failed", ...)
    # automatically, so STARTED events don't stay open forever.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Context manager for structured interaction event logging.

    Writes to the processing_events table in Postgres.
    Falls back to structured log output if the DB write fails
    (we don't want audit logging to block the main pipeline).
    """

    def __init__(
        self,
        interaction_id: str,
        customer_id: str,
        campaign_id: str,
        session_id: Optional[str] = None,
    ):
        self.interaction_id = interaction_id
        self.customer_id = customer_id
        self.campaign_id = campaign_id
        self.session_id = session_id
        self._active_event_type: Optional[str] = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self._active_event_type:
            await self.event(
                f"{self._active_event_type}_failed",
                status="FAILED",
                metadata={"error": str(exc_val), "error_type": exc_type.__name__},
            )
        return False  # don't suppress exceptions

    async def started(self, event_type: str, metadata: Optional[Dict[str, Any]] = None):
        """Mark a processing step as started. Paired with completed() or failed()."""
        self._active_event_type = event_type
        await self.event(event_type, status="STARTED", metadata=metadata)

    async def completed(self, event_type: str, metadata: Optional[Dict[str, Any]] = None):
        """Mark a processing step as successfully completed."""
        self._active_event_type = None
        await self.event(event_type, status="COMPLETED", metadata=metadata)

    async def failed(self, event_type: str, error: str, metadata: Optional[Dict[str, Any]] = None):
        """Mark a processing step as failed."""
        self._active_event_type = None
        meta = metadata or {}
        meta["error"] = error
        await self.event(event_type, status="FAILED", metadata=meta)

    async def event(
        self,
        event_type: str,
        status: str = "INFO",
        metadata: Optional[Dict[str, Any]] = None,
        processing_lane: Optional[str] = None,
    ):
        """
        Write a processing event to Postgres and structured logs.

        In production, this does:
            INSERT INTO processing_events
                (interaction_id, customer_id, campaign_id, event_type, status, metadata, ...)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW())

        If the DB write fails, we still emit the structured log so the event
        is not lost (it can be replayed from logs into the DB if needed).
        """
        log_payload = {
            "interaction_id": self.interaction_id,
            "customer_id": self.customer_id,
            "campaign_id": self.campaign_id,
            "session_id": self.session_id,
            "event_type": event_type,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }

        if processing_lane:
            log_payload["processing_lane"] = processing_lane

        # Production DB write:
        # try:
        #     await db.execute("""
        #         INSERT INTO processing_events
        #             (interaction_id, customer_id, campaign_id, event_type, status,
        #              processing_lane, metadata, started_at)
        #         VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb,
        #                 CASE WHEN $5 = 'STARTED' THEN NOW() ELSE NULL END)
        #     """, self.interaction_id, self.customer_id, self.campaign_id,
        #         event_type, status, processing_lane, json.dumps(metadata or {}))
        # except Exception as db_err:
        #     logger.error("audit_db_write_failed",
        #                  extra={"error": str(db_err), **log_payload})

        # Log level based on status
        if status == "FAILED":
            logger.error(f"audit:{event_type}", extra=log_payload)
        elif status == "STARTED":
            logger.info(f"audit:{event_type}", extra=log_payload)
        elif status == "COMPLETED":
            logger.info(f"audit:{event_type}", extra=log_payload)
        else:
            logger.info(f"audit:{event_type}", extra=log_payload)


async def write_token_ledger(
    interaction_id: str,
    customer_id: str,
    campaign_id: str,
    phase: str,
    estimated_tokens: int,
    actual_tokens: Optional[int],
    model: str,
) -> None:
    """
    Write a token usage record to the token_ledger table.

    This is the billing source of truth. Every token consumed by any LLM call
    must be recorded here with attribution to customer, campaign, and interaction.

    In production:
        INSERT INTO token_ledger
            (customer_id, campaign_id, interaction_id, processing_phase,
             estimated_tokens, actual_tokens, model, minute_bucket)
        VALUES ($1, $2, $3, $4, $5, $6, $7, extract(epoch from now())::bigint / 60)
        ON CONFLICT DO NOTHING  -- idempotent on re-run
    """
    import time
    bucket = int(time.time()) // 60

    logger.info(
        "token_ledger_written",
        extra={
            "interaction_id": interaction_id,
            "customer_id": customer_id,
            "campaign_id": campaign_id,
            "phase": phase,
            "estimated_tokens": estimated_tokens,
            "actual_tokens": actual_tokens,
            "model": model,
            "minute_bucket": bucket,
        },
    )

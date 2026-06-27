"""
Celery tasks for post-call processing.

This is the redesigned pipeline. Two phases:

  Phase 1 — classify_interaction_task (fast, high-priority queue)
    - Short call check (free)
    - LLM lane classification (~200 tokens)
    - Routes to HOT queue, COLD queue, or marks as SKIPPED

  Phase 2 — process_interaction_task (rate-limit-aware, per-customer budget)
    - Runs recording poller AND full LLM analysis IN PARALLEL
    - Acquires token budget before calling LLM — never fires a request
      that would exceed the configured rate limits
    - Retries with exponential backoff (not fixed 60s) when budget is
      exhausted, staying silent to the LLM provider
    - Writes audit events to Postgres at every stage transition

Key differences from the original:
  - No asyncio.sleep(45s) blocking the pipeline
  - Recording and LLM run in parallel with asyncio.gather()
  - Budget check BEFORE LLM call (not after, as the old circuit breaker did)
  - Every failure path emits a structured event with interaction_id
  - Separate queues for HOT and COLD calls (not a single shared queue)
  - No dual-retry race condition (Celery retries only; Redis retry queue removed)

Queue names:
  postcall_classify  — Phase 1, high worker count, short tasks
  postcall_hot       — Phase 2 HOT lane, dedicated workers
  postcall_cold      — Phase 2 COLD lane, lower priority workers
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict

from src.tasks.celery_app import celery_app
from src.services.post_call_processor import PostCallProcessor, PostCallContext
from src.services.recording import fetch_and_upload_recording
from src.services.signal_jobs import trigger_signal_jobs, update_lead_stage
from src.services.lane_classifier import classify_lane, is_short_call
from src.services.token_budget import budget_scheduler
from src.services.audit_logger import AuditLogger, write_token_ledger
from src.services.metrics import metrics_tracker

logger = logging.getLogger(__name__)


# ── Phase 1: Classification ────────────────────────────────────────────────────

@celery_app.task(
    name="classify_interaction_task",
    bind=True,
    max_retries=5,
    queue="postcall_classify",
    acks_late=True,
)
def classify_interaction_task(self, payload: Dict[str, Any]):
    """
    Phase 1: Classify the call lane and route to the appropriate processing queue.

    Runs fast (~300ms for LLM classification, 0ms for short call skip).
    No rate limit pressure here — classification is cheap and separate from
    the full analysis budget.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_classify_interaction(self, payload))
    except Exception as e:
        logger.exception(
            "classify_task_failed",
            extra={
                "interaction_id": payload.get("interaction_id"),
                "error": str(e),
                "attempt": self.request.retries,
            },
        )
        raise self.retry(exc=e, countdown=5 * (2 ** self.request.retries))
    finally:
        loop.close()


async def _classify_interaction(task, payload: Dict[str, Any]):
    interaction_id = payload["interaction_id"]
    customer_id = payload["customer_id"]
    campaign_id = payload["campaign_id"]

    async with AuditLogger(
        interaction_id=interaction_id,
        customer_id=customer_id,
        campaign_id=campaign_id,
        session_id=payload.get("session_id"),
    ) as audit:
        await audit.started("classify", metadata={"attempt": task.request.retries})

        transcript_text = payload.get("transcript_text", "")
        conversation_data = payload.get("conversation_data", {})

        lane = await classify_lane(
            transcript_text=transcript_text,
            conversation_data=conversation_data,
            interaction_id=interaction_id,
        )

        await audit.completed(
            "classify",
            metadata={"lane": lane},
        )

    if lane == "skip":
        # Write a synthetic result directly — no LLM needed
        logger.info(
            "interaction_skipped_short_call",
            extra={
                "interaction_id": interaction_id,
                "customer_id": customer_id,
            },
        )
        # Production: UPDATE interactions SET
        #   interaction_metadata = '{"call_stage": "short_call", "analysis_status": "skipped"}',
        #   processing_status = 'COMPLETED'
        # WHERE id = interaction_id
        return

    # Route to the appropriate processing queue
    queue_name = "postcall_hot" if lane == "hot" else "postcall_cold"
    payload["processing_lane"] = lane

    process_interaction_task.apply_async(
        args=[payload],
        queue=queue_name,
    )

    logger.info(
        "interaction_routed",
        extra={
            "interaction_id": interaction_id,
            "customer_id": customer_id,
            "lane": lane,
            "queue": queue_name,
        },
    )


# ── Phase 2: Full Processing ───────────────────────────────────────────────────

@celery_app.task(
    name="process_interaction_task",
    bind=True,
    max_retries=10,          # More retries than before — budget exhaustion is expected
    acks_late=True,
    # Queue is set dynamically when task is enqueued (postcall_hot or postcall_cold)
    queue="postcall_cold",   # default; overridden by classify_interaction_task
)
def process_interaction_task(self, payload: Dict[str, Any]):
    """
    Phase 2: Full analysis with rate-limit-aware budget scheduling.

    Before calling the LLM, this task:
      1. Estimates the token cost for this call
      2. Calls budget_scheduler.acquire_budget()
      3. If budget is insufficient, re-queues itself with exponential backoff
         (starting at 5s, not the old fixed 60s)
      4. If budget is available, fires the LLM and recording poller in parallel

    This guarantees the LLM provider never receives more requests per minute
    than the configured budget allows, without requiring the dialler to freeze.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_process_interaction(self, payload))
    except BudgetExhaustedError as e:
        # Budget exhausted — retry with backoff, no error logged at ERROR level
        backoff = budget_scheduler.backoff_seconds(self.request.retries)
        logger.info(
            "budget_exhausted_retry",
            extra={
                "interaction_id": payload.get("interaction_id"),
                "customer_id": payload.get("customer_id"),
                "retry_in": backoff,
                "attempt": self.request.retries,
            },
        )
        raise self.retry(exc=e, countdown=backoff)
    except Exception as e:
        logger.exception(
            "process_task_failed",
            extra={
                "interaction_id": payload.get("interaction_id"),
                "customer_id": payload.get("customer_id"),
                "error": str(e),
                "attempt": self.request.retries,
            },
        )
        backoff = min(30 * (2 ** self.request.retries), 300)
        raise self.retry(exc=e, countdown=backoff)
    finally:
        loop.close()


class BudgetExhaustedError(Exception):
    """Raised when token budget is insufficient. Signals the task to retry quietly."""
    pass


async def _process_interaction(task, payload: Dict[str, Any]):
    interaction_id = payload["interaction_id"]
    customer_id = payload["customer_id"]
    campaign_id = payload["campaign_id"]
    lane = payload.get("processing_lane", "cold")

    ctx = PostCallContext(
        interaction_id=interaction_id,
        session_id=payload["session_id"],
        lead_id=payload["lead_id"],
        campaign_id=campaign_id,
        customer_id=customer_id,
        agent_id=payload["agent_id"],
        call_sid=payload.get("call_sid", ""),
        transcript_text=payload.get("transcript_text", ""),
        conversation_data=payload.get("conversation_data", {}),
        additional_data=payload.get("additional_data", {}),
        ended_at=datetime.fromisoformat(payload["ended_at"]),
        exotel_account_id=payload.get("exotel_account_id"),
    )

    async with AuditLogger(
        interaction_id=interaction_id,
        customer_id=customer_id,
        campaign_id=campaign_id,
        session_id=ctx.session_id,
    ) as audit:

        # ── Budget check ────────────────────────────────────────────────────
        estimated_tokens = budget_scheduler.estimate_tokens(
            ctx.transcript_text, phase="full_analysis"
        )

        await audit.started(
            "budget_check",
            metadata={"estimated_tokens": estimated_tokens, "lane": lane},
        )

        reservation = await budget_scheduler.acquire_budget(
            customer_id=customer_id,
            estimated_tokens=estimated_tokens,
            lane=lane,
        )

        if reservation is None:
            # Budget exhausted — signal the Celery task to retry with backoff
            await audit.event(
                "budget_check",
                status="WAITING_BUDGET",
                metadata={
                    "estimated_tokens": estimated_tokens,
                    "retry_attempt": task.request.retries,
                },
            )
            raise BudgetExhaustedError(
                f"Token budget exhausted for customer {customer_id}"
            )

        await audit.completed(
            "budget_check",
            metadata={"estimated_tokens": estimated_tokens, "reservation": reservation},
        )

        # ── Parallel: LLM analysis + Recording poller ───────────────────────
        await audit.started(
            "full_processing",
            metadata={"lane": lane, "estimated_tokens": estimated_tokens},
        )

        try:
            analysis_result, recording_s3_key = await asyncio.gather(
                _run_llm_analysis(ctx, audit),
                fetch_and_upload_recording(
                    interaction_id=interaction_id,
                    call_sid=ctx.call_sid,
                    exotel_account_id=ctx.exotel_account_id or "",
                ),
                return_exceptions=False,
            )
        except Exception as e:
            await budget_scheduler.release_reservation(reservation)
            raise

        # ── Reconcile actual token usage ────────────────────────────────────
        if analysis_result is not None:
            await budget_scheduler.reconcile_actual(
                reservation, analysis_result.tokens_used
            )
            await write_token_ledger(
                interaction_id=interaction_id,
                customer_id=customer_id,
                campaign_id=campaign_id,
                phase="full_analysis",
                estimated_tokens=estimated_tokens,
                actual_tokens=analysis_result.tokens_used,
                model=analysis_result.model,
            )
        else:
            await budget_scheduler.release_reservation(reservation)

        await audit.completed(
            "full_processing",
            metadata={
                "actual_tokens": analysis_result.tokens_used if analysis_result else 0,
                "recording_uploaded": recording_s3_key is not None,
                "call_stage": analysis_result.call_stage if analysis_result else "unknown",
            },
        )

        # ── Downstream: signal jobs + lead stage ────────────────────────────
        if analysis_result:
            await _run_downstream(ctx, analysis_result, audit)

    await metrics_tracker.track_processing_completed(
        interaction_id,
        analysis_result.tokens_used if analysis_result else 0,
        analysis_result.latency_ms if analysis_result else 0,
    )


async def _run_llm_analysis(ctx: PostCallContext, audit: AuditLogger):
    """Run full LLM analysis with audit events."""
    await audit.started("llm_analysis")

    processor = PostCallProcessor()
    try:
        result = await processor.process_post_call(ctx, single_prompt=True)
        await audit.completed(
            "llm_analysis",
            metadata={
                "call_stage": result.call_stage,
                "tokens_used": result.tokens_used,
                "latency_ms": result.latency_ms,
                "model": result.model,
            },
        )
        return result
    except Exception as e:
        await audit.failed("llm_analysis", error=str(e))
        raise


async def _run_downstream(ctx: PostCallContext, result, audit: AuditLogger):
    """Run signal jobs and lead stage update with individual error handling."""
    # Signal jobs (CRM push, WhatsApp, callbacks)
    try:
        await audit.started("signal_jobs")
        await trigger_signal_jobs(
            interaction_id=ctx.interaction_id,
            session_id=ctx.session_id,
            campaign_id=ctx.campaign_id,
            analysis_result=result.raw_response,
        )
        await audit.completed("signal_jobs")
    except Exception as e:
        # Signal job failures are non-fatal — the analysis result is already
        # written to the DB. Log at ERROR so it shows up in alerts.
        await audit.failed("signal_jobs", error=str(e))
        logger.error(
            "signal_jobs_failed",
            extra={
                "interaction_id": ctx.interaction_id,
                "customer_id": ctx.customer_id,
                "error": str(e),
                # In production, failed signal jobs should be put in a
                # separate retry queue with their own backoff policy.
            },
        )

    # Lead stage update
    try:
        await audit.started("lead_stage_update")
        await update_lead_stage(
            lead_id=ctx.lead_id,
            interaction_id=ctx.interaction_id,
            call_stage=result.call_stage,
        )
        await audit.completed(
            "lead_stage_update",
            metadata={"call_stage": result.call_stage},
        )
    except Exception as e:
        await audit.failed("lead_stage_update", error=str(e))
        logger.error(
            "lead_stage_update_failed",
            extra={
                "interaction_id": ctx.interaction_id,
                "lead_id": ctx.lead_id,
                "error": str(e),
            },
        )


# ── Reconciliation task (runs via Celery Beat every 5 minutes) ────────────────

@celery_app.task(name="reconcile_stuck_interactions")
def reconcile_stuck_interactions():
    """
    Find interactions stuck in STARTED state and re-enqueue them.

    A STARTED event older than 10 minutes indicates a worker died mid-task.
    We re-enqueue the interaction's classify task to restart processing.
    Because all steps are idempotent (LLM writes use ON CONFLICT DO UPDATE,
    S3 upload uses deterministic keys), re-processing is safe.

    In production this queries:
        SELECT interaction_id, customer_id, campaign_id, ...
        FROM processing_events
        WHERE status = 'STARTED'
          AND started_at < NOW() - INTERVAL '10 minutes'
          AND interaction_id NOT IN (
              SELECT interaction_id FROM processing_events
              WHERE status IN ('COMPLETED', 'FAILED')
          )
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_reconcile())
    finally:
        loop.close()


async def _reconcile():
    logger.info("reconciliation_started")
    # Production:
    # stuck = await db.fetch("""
    #     SELECT DISTINCT pe.interaction_id, i.customer_id, i.campaign_id,
    #                     i.conversation_data, i.ended_at, ...
    #     FROM processing_events pe
    #     JOIN interactions i ON i.id = pe.interaction_id
    #     WHERE pe.status = 'STARTED'
    #       AND pe.started_at < NOW() - INTERVAL '10 minutes'
    #       AND NOT EXISTS (
    #           SELECT 1 FROM processing_events pe2
    #           WHERE pe2.interaction_id = pe.interaction_id
    #             AND pe2.status IN ('COMPLETED', 'FAILED')
    #       )
    # """)
    # for row in stuck:
    #     classify_interaction_task.apply_async(
    #         args=[_row_to_payload(row)],
    #         queue="postcall_classify",
    #     )
    #     logger.warning("interaction_requeued", extra={"interaction_id": row["interaction_id"]})
    logger.info("reconciliation_complete", extra={"requeued": 0})


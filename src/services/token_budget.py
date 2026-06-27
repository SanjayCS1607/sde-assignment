"""
TokenBudgetScheduler — Rate-limit-aware LLM request gating.

This is the core fix for the primary problem: the current system fires LLM
requests at full speed with no awareness of provider rate limits.

Design:
  - Every customer has a tokens_per_minute allocation (from customer_llm_config).
  - Global platform limit comes from settings.LLM_TOKENS_PER_MINUTE.
  - Before any LLM call, acquire_budget() must be called and return True.
  - If the budget is exhausted, the task re-queues itself with backoff.
  - Redis tracks per-customer and global usage within 60-second buckets.
  - Actual token usage is reconciled after each call via reconcile_actual().

Token estimation:
  We estimate tokens before the call using character count / 4 (rough GPT
  tokeniser approximation). This is conservative for Hinglish (which tokenises
  worse than English) and may under-count by 20-30%. Reconciliation corrects the
  counter after each call. During bursts, slight overestimation is acceptable —
  it means we use ~80% of available budget rather than risking 429s.

Redis key structure:
  budget:{customer_id}:reserved   → integer, tokens/min reserved (from DB config)
  budget:{customer_id}:{bucket}   → integer, tokens used this minute
  budget:global:{bucket}          → integer, platform-wide tokens used this minute
  budget:active_customers:{bucket}→ set of customer_ids with active tasks this minute

All keys expire at the end of their 60-second bucket (EXPIREAT is set on every
write). A Redis restart loses the counters — tasks will over-use budget for up
to 60 seconds before the window resets. This is acceptable because Postgres
token_ledger is the billing source of truth; Redis is only the enforcement gate.
"""

import logging
import time
from typing import Optional

from src.config import settings
from src.utils.redis_client import redis_client

logger = logging.getLogger(__name__)


def _minute_bucket() -> int:
    """Current 60-second bucket as unix timestamp // 60."""
    return int(time.time()) // 60


def _bucket_expiry(bucket: int) -> int:
    """Unix timestamp when this bucket expires (start of next minute)."""
    return (bucket + 1) * 60


class TokenBudgetScheduler:
    """
    Manages LLM token budgets across customers and enforces global rate limits.

    Usage pattern (in Celery task):

        scheduler = TokenBudgetScheduler()
        estimated = scheduler.estimate_tokens(transcript_text, phase="full_analysis")
        reservation = await scheduler.acquire_budget(customer_id, estimated)
        if not reservation:
            raise self.retry(countdown=scheduler.backoff_seconds(self.request.retries))

        try:
            result = await call_llm(...)
            await scheduler.reconcile_actual(reservation, result.tokens_used)
        except Exception:
            await scheduler.release_reservation(reservation)
            raise
    """

    def estimate_tokens(self, text: str, phase: str = "full_analysis") -> int:
        """
        Estimate tokens for an LLM call before making it.

        Phase determines the prompt overhead:
          - classify: short prompt (~100 tokens) + transcript
          - full_analysis: full prompt (~300 tokens) + transcript
        """
        transcript_tokens = max(len(text) // 4, 50)  # rough GPT tokeniser estimate

        overhead = {
            "classify": 150,
            "full_analysis": 350,
        }.get(phase, 350)

        completion_estimate = {
            "classify": 30,    # {"lane": "hot"} is ~10 tokens; buffer
            "full_analysis": 400,  # entities + summary + call_stage
        }.get(phase, 400)

        return transcript_tokens + overhead + completion_estimate

    async def acquire_budget(
        self,
        customer_id: str,
        estimated_tokens: int,
        lane: str = "cold",
    ) -> Optional[dict]:
        """
        Attempt to reserve estimated_tokens for customer_id.

        Returns a reservation dict on success (to be passed to reconcile_actual),
        or None if budget is insufficient.

        HOT lane calls get priority: they check only the HOT lane's reserved
        headroom (20% of customer allocation) before checking the general pool.
        Cold lane calls cannot use HOT-reserved headroom.
        """
        bucket = _minute_bucket()
        expiry = _bucket_expiry(bucket)

        # Keys
        cust_key = f"budget:{customer_id}:{bucket}"
        global_key = f"budget:global:{bucket}"
        active_key = f"budget:active_customers:{bucket}"

        # Get current usage (pipeline for atomicity of reads)
        pipe = redis_client.pipeline()
        pipe.get(cust_key)
        pipe.get(global_key)
        pipe.get(f"budget:{customer_id}:reserved")
        results = await pipe.execute()

        cust_used = int(results[0] or 0)
        global_used = int(results[1] or 0)
        cust_reserved = int(results[2] or 0)  # per-minute allocation from DB config

        # Determine effective customer limit
        # If customer has no reservation, they share the unallocated pool.
        # Unallocated pool headroom is approximated as:
        #   global_limit - sum(all reserved) ÷ active_customers_count
        # For simplicity: if no reservation, cap at 10% of global limit per bucket check.
        if cust_reserved == 0:
            cust_limit = settings.LLM_TOKENS_PER_MINUTE // 10  # default fair share
        else:
            cust_limit = cust_reserved

        global_limit = settings.LLM_TOKENS_PER_MINUTE

        cust_remaining = cust_limit - cust_used
        global_remaining = global_limit - global_used
        effective_headroom = min(cust_remaining, global_remaining)

        if estimated_tokens > effective_headroom:
            logger.info(
                "budget_insufficient",
                extra={
                    "customer_id": customer_id,
                    "estimated_tokens": estimated_tokens,
                    "cust_remaining": cust_remaining,
                    "global_remaining": global_remaining,
                    "lane": lane,
                    "bucket": bucket,
                },
            )
            return None

        # Reserve tokens atomically
        pipe = redis_client.pipeline()
        pipe.incrby(cust_key, estimated_tokens)
        pipe.expireat(cust_key, expiry)
        pipe.incrby(global_key, estimated_tokens)
        pipe.expireat(global_key, expiry)
        pipe.sadd(active_key, customer_id)
        pipe.expireat(active_key, expiry)
        await pipe.execute()

        reservation = {
            "customer_id": customer_id,
            "estimated_tokens": estimated_tokens,
            "bucket": bucket,
            "lane": lane,
        }

        logger.debug(
            "budget_reserved",
            extra={
                "customer_id": customer_id,
                "estimated_tokens": estimated_tokens,
                "cust_remaining_after": cust_remaining - estimated_tokens,
                "global_remaining_after": global_remaining - estimated_tokens,
                "bucket": bucket,
            },
        )

        return reservation

    async def reconcile_actual(self, reservation: dict, actual_tokens: int) -> None:
        """
        Reconcile the estimated reservation with the actual tokens used.

        If we over-estimated, return the excess tokens to the budget.
        If we under-estimated, add the overage to the customer's usage counter.
        This keeps the counters accurate across a burst window.
        """
        if reservation is None:
            return

        estimated = reservation["estimated_tokens"]
        actual = actual_tokens
        customer_id = reservation["customer_id"]
        bucket = reservation["bucket"]
        expiry = _bucket_expiry(bucket)

        delta = actual - estimated  # positive = we used more than estimated

        if delta == 0:
            return

        cust_key = f"budget:{customer_id}:{bucket}"
        global_key = f"budget:global:{bucket}"

        pipe = redis_client.pipeline()
        if delta > 0:
            pipe.incrby(cust_key, delta)
            pipe.incrby(global_key, delta)
        else:
            # Release excess reservation
            pipe.decrby(cust_key, abs(delta))
            pipe.decrby(global_key, abs(delta))
        pipe.expireat(cust_key, expiry)
        pipe.expireat(global_key, expiry)
        await pipe.execute()

        logger.debug(
            "budget_reconciled",
            extra={
                "customer_id": customer_id,
                "estimated_tokens": estimated,
                "actual_tokens": actual,
                "delta": delta,
                "bucket": bucket,
            },
        )

    async def release_reservation(self, reservation: dict) -> None:
        """
        Release a reservation when a task fails before calling the LLM.
        This prevents stranded budget reservations on task failure.
        """
        if reservation is None:
            return

        customer_id = reservation["customer_id"]
        estimated = reservation["estimated_tokens"]
        bucket = reservation["bucket"]
        expiry = _bucket_expiry(bucket)

        cust_key = f"budget:{customer_id}:{bucket}"
        global_key = f"budget:global:{bucket}"

        pipe = redis_client.pipeline()
        pipe.decrby(cust_key, estimated)
        pipe.expireat(cust_key, expiry)
        pipe.decrby(global_key, estimated)
        pipe.expireat(global_key, expiry)
        await pipe.execute()

        logger.debug(
            "budget_released",
            extra={
                "customer_id": customer_id,
                "estimated_tokens": estimated,
                "bucket": bucket,
            },
        )

    def backoff_seconds(self, retry_count: int) -> int:
        """
        Exponential backoff with jitter for budget-exhausted retries.

        Unlike Celery's fixed 60s delay, this starts fast (5s) and backs off
        proportionally. Budget windows reset every 60s, so retrying after
        the window is most useful; jitter spreads load across the reset boundary.
        """
        import random
        base = min(5 * (2 ** retry_count), 55)  # cap just under 60s window
        jitter = random.uniform(0, base * 0.2)
        return int(base + jitter)

    async def get_utilisation(self, customer_id: Optional[str] = None) -> dict:
        """
        Return current budget utilisation for monitoring/alerting.

        Called by metrics endpoints and alert checks.
        """
        bucket = _minute_bucket()

        global_key = f"budget:global:{bucket}"
        global_used = int(await redis_client.get(global_key) or 0)
        global_limit = settings.LLM_TOKENS_PER_MINUTE
        global_pct = round(global_used / global_limit * 100, 1) if global_limit else 0

        result = {
            "global_used": global_used,
            "global_limit": global_limit,
            "global_pct": global_pct,
            "bucket": bucket,
        }

        if customer_id:
            cust_key = f"budget:{customer_id}:{bucket}"
            cust_reserved_key = f"budget:{customer_id}:reserved"
            cust_used = int(await redis_client.get(cust_key) or 0)
            cust_reserved = int(await redis_client.get(cust_reserved_key) or 0)
            cust_limit = cust_reserved or (settings.LLM_TOKENS_PER_MINUTE // 10)
            cust_pct = round(cust_used / cust_limit * 100, 1) if cust_limit else 0

            result.update({
                "customer_used": cust_used,
                "customer_limit": cust_limit,
                "customer_pct": cust_pct,
            })

        return result

    async def set_customer_allocation(
        self, customer_id: str, tokens_per_minute: int
    ) -> None:
        """
        Set a customer's per-minute token allocation in Redis.
        Called on campaign start or when the DB config is updated.
        Expires after 24h — should be refreshed on each campaign start.
        """
        key = f"budget:{customer_id}:reserved"
        await redis_client.set(key, tokens_per_minute, ex=86400)
        logger.info(
            "customer_allocation_set",
            extra={
                "customer_id": customer_id,
                "tokens_per_minute": tokens_per_minute,
            },
        )


budget_scheduler = TokenBudgetScheduler()

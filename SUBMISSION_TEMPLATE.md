# Post-Call Processing Pipeline — Design Document

**Author:** Sanjay C S 
**Date:** 27-06-2026

---

## 1. Assumptions

1. **Campaign window is time-bounded.** A campaign run typically completes within 4–8 hours. Processing latency beyond that window reduces business value — a "rebook_confirmed" that gets analysed 12 hours after the call is useful, but one processed within minutes enables same-day follow-up.

2. **"Hot" outcomes are determinable cheaply.** The sample transcripts confirm that high-value dispositions (`rebook_confirmed`, `demo_booked`, `escalation_needed`) can be identified with a lightweight prompt (~200 tokens) before running the full 1,500-token analysis. A two-phase approach is economically justified.

3. **LLM rate limits are shared across all customers on a single API key.** The current config has one `LLM_API_KEY`. Per-customer token budgets are enforced at the application layer, not at the provider.

4. **Exotel recordings are independent of the transcript.** The LLM reads the transcript text; recording upload is a storage-only operation. They can run in parallel and failure in one should not block the other.

5. **Redis is available but not durable.** Redis is used for rate-limit counters and caching. The source of truth for all state lives in Postgres. Any Redis value can be reconstructed from Postgres if lost.

6. **"Permanent loss" means loss from the DB.** A task that fails and is re-queued is not lost. A result that is never written to Postgres and not re-queued is permanently lost — this is what we must prevent.

7. **Customer priority is a first-class config, not a runtime signal.** Customers can have a pre-allocated `tokens_per_minute` budget set at campaign creation. A customer without a pre-allocation gets a share of the unallocated pool.

8. **The Celery + Redis broker combination is retained for now.** Replacing it with Postgres-backed durability (e.g., pg-boss, Temporal) would be the right long-term move, but the migration is out of scope. Instead we make Celery tasks idempotent and add a Postgres-backed durability layer so broker restarts cannot cause permanent loss.

9. **Short call threshold is <4 turns** (matching the existing system). Short calls skip LLM entirely and get a synthetic `short_call` disposition written directly to Postgres.

10. **Token estimation is pre-call.** We estimate tokens before firing the LLM using `len(transcript) // 4` (rough GPT tokeniser estimate). Actual usage is reconciled after the call and written to the per-customer budget ledger.

---

## 2. Problem Diagnosis

The system has five compounding failure modes:

**1. No rate-limit awareness.** `LLM_TOKENS_PER_MINUTE` exists in config but nothing reads it. At 100K calls, the system fires all LLM requests at full speed, gets 429s, and then blindly retries after a fixed 60s — causing the retry storm to exceed the original load.

**2. Sequential blocking on recording.** `asyncio.sleep(45s)` blocks LLM analysis even though they share zero dependencies. Under load this wastes 45 × (number of concurrent tasks) seconds of worker time before the queue even starts draining.

**3. No per-customer fairness.** One customer's campaign can consume the entire LLM token budget, delaying or starving other customers. There is no isolation.

**4. Fragile durability.** Tasks are in-flight in Redis. A broker restart drops them. The retry queue is also in Redis. A single Redis failure loses both the task and the retry record. No Postgres audit trail means there's no way to know what was lost.

**5. Binary circuit breaker.** 89% capacity: go. 90% capacity: freeze all diallers for 30 minutes. The correct behaviour is proportional backpressure, not a full stop.

---

## 3. Architecture Overview

```
POST /session/{sid}/interaction/{iid}/end
        │
        ▼
   FastAPI endpoint
   ├── Validate payload
   ├── Write interaction row to Postgres: status=QUEUED, processing_lane=TBD
   └── Enqueue: classify_interaction_task (Celery, high-priority queue)
                │
                ▼
        ┌───────────────────────────────────┐
        │  Phase 1: Cheap Classification    │  ~200 tokens, ~300ms
        │  - Short call? → write short_call │
        │  - Otherwise: classify_lane()     │
        │    → "hot" or "cold"              │
        └───────────────┬───────────────────┘
                        │
           ┌────────────┴────────────┐
           │ hot                     │ cold
           ▼                         ▼
   ┌───────────────┐        ┌─────────────────┐
   │ HOT queue     │        │ COLD/BATCH queue │
   │ TokenBudget   │        │ TokenBudget      │
   │ Scheduler     │        │ Scheduler        │
   └──────┬────────┘        └────────┬─────────┘
          │                          │
          └──────────┬───────────────┘
                     ▼
          ┌──────────────────────────────┐
          │  Full Analysis Task          │
          │  (run in parallel):          │
          │  ├── LLM: full analysis      │
          │  │   (entities, summary,     │
          │  │    call_stage, tokens)    │
          │  └── Recording poller        │
          │      (poll Exotel w/backoff) │
          └──────────┬───────────────────┘
                     ▼
          ┌──────────────────────────────┐
          │  Downstream Tasks            │
          │  ├── Write interaction_metadata│
          │  ├── Update lead stage        │
          │  ├── Write token_ledger row   │
          │  └── Signal jobs (CRM, WA)   │
          └──────────────────────────────┘
```

### Key design decisions

1. **Two-phase processing.** Phase 1 (cheap classification: ~200 tokens, ~300ms) routes calls to HOT or COLD queues. Phase 2 (full analysis: ~1,500 tokens) is scheduled by the token budget scheduler. This decouples "which calls matter now" from "when we have capacity to analyse them."

2. **Token Budget Scheduler per customer.** Each customer has a `tokens_per_minute` allocation (stored in Postgres). A Redis sorted set tracks remaining budget per customer per 60s window. Tasks are dequeued only when budget headroom exists. Unallocated headroom is divided among active customers in proportion to their allocation ("work-conserving" scheduling).

3. **Postgres-backed audit trail.** Before a task runs, a `processing_events` row is written with `status=STARTED`. On completion, it is updated to `COMPLETED`. On failure, `FAILED` with the error. If a worker dies mid-task, the row stays at `STARTED` and a reconciliation job re-enqueues it. Redis broker loss never causes permanent loss.

4. **Parallel recording + LLM.** `asyncio.gather()` runs recording polling and LLM analysis concurrently. Recording failure does not affect analysis result.

5. **Proportional backpressure instead of circuit breaker.** When token budget utilisation for a customer exceeds 80%, their classification tasks slow down (add jitter). No binary freeze. The dialler is not directly involved — it naturally dispatches fewer calls when post-call processing produces delayed "available" signals.

---

## 4. Rate Limit Management

### How we track rate limit usage

```
Redis key: budget:{customer_id}:used_tokens:{minute_bucket}
  - minute_bucket = unix_timestamp // 60
  - Value: cumulative tokens used this minute (INCRBY + EXPIREAT end of minute)
  - Estimated before call, reconciled with actual after call
```

A global key `budget:global:used_tokens:{minute_bucket}` tracks platform-wide usage.

### How we decide what to process now vs. defer

The `TokenBudgetScheduler` is called before every LLM task:

```python
async def acquire_budget(customer_id: str, estimated_tokens: int) -> bool:
    customer_remaining = customer_allocation - customer_used
    global_remaining = global_limit - global_used
    headroom = min(customer_remaining, global_remaining)
    
    if estimated_tokens <= headroom:
        # Reserve the tokens optimistically
        await redis.incrby(f"budget:{customer_id}:used_tokens:{bucket}", estimated_tokens)
        await redis.incrby(f"budget:global:used_tokens:{bucket}", estimated_tokens)
        return True
    return False  # Task re-queued with delay
```

When `acquire_budget` returns False, the task is retried with exponential backoff starting at 5s (not the current fixed 60s). The retry count and delay are visible in the `processing_events` table.

### What happens when the limit is hit (recovery, not crash)

- `acquire_budget()` returns False → task re-queues itself with `countdown=backoff_seconds`
- No 429 is surfaced to any caller
- The `processing_events` row is updated to `status=WAITING_BUDGET` so dashboards show the queue state
- When the minute rolls over, budget resets and queued tasks drain

---

## 5. Per-Customer Token Budgeting

### Allocation model

Stored in a new `customer_llm_config` table:

| customer_id | tokens_per_minute_reserved | priority |
|-------------|---------------------------|----------|
| cust_a      | 20,000                    | high     |
| cust_b      | 15,000                    | standard |
| (unresolved) | 55,000 pool              | —        |

- **Reserved:** Guaranteed minimum — Customer A always gets 20,000 TPM even if Customer B has a massive campaign.
- **Unallocated pool:** 55,000 TPM shared among all customers without reservations. Divided by the number of active customers (those with tasks in-flight this minute).
- **Burst:** A customer may use above their reserved amount IF there is unallocated headroom AND no other customer is being starved. Burst tokens are counted against the customer's budget for billing but do not reduce other customers' guaranteed minimums.

### What happens when a customer exceeds their budget

- Tasks are re-queued with `status=WAITING_BUDGET` — visible in the audit table
- No 429 from the provider (we never send the request)
- HOT-lane tasks for that customer are still prioritised over their own COLD-lane tasks — the customer's budget is enforced as a shared pool, not segregated by lane

### What happens to unallocated headroom

Work-conserving: unallocated headroom is distributed proportionally to customers with active queues every 5 seconds. No headroom is ever wasted while there are tasks to process.

---

## 6. Differentiated Processing

### Classification approach

A lightweight Phase 1 LLM call (~200 tokens) with a tight prompt:

```
Classify this call transcript into one of: hot, cold, skip.
hot = booking confirmed, demo scheduled, escalation needed, payment taken
cold = callback requested, considering, not interested, already done
skip = less than 4 turns, wrong number, no meaningful exchange
Return only: {"lane": "hot"|"cold"|"skip"}
```

**Why LLM and not keyword rules?** The Hinglish transcript ("Dekhta hoon, kuch sochna padega") shows that keyword matching fails on multilingual, colloquial speech. An LLM classifier handles this; keyword rules would misclassify it.

**Why a two-phase approach and not one bigger prompt?** Phase 1 costs ~200 tokens. On 100K calls, saving 1,300 tokens for cold/skip calls saves 130M tokens/campaign — at $0.01/1K tokens that's $1,300 per campaign. More importantly, it lets hot calls jump the queue without waiting for the full analysis budget.

### Lane routing

- `skip` → no LLM, write `{call_stage: "short_call", analysis_status: "skipped"}` directly, mark done
- `hot` → HOT Celery queue (separate queue, higher worker count)
- `cold` → COLD Celery queue (can tolerate 5–30 minute processing delay)

HOT queue workers have dedicated budget allocation — when total platform budget is constrained, HOT tasks always get first access.

---

## 7. Recording Pipeline

Replacing `asyncio.sleep(45s)`:

```python
async def poll_recording(interaction_id, call_sid, account_id) -> Optional[str]:
    max_attempts = 8
    for attempt in range(max_attempts):
        url = await _fetch_exotel_recording_url(call_sid, account_id)
        if url:
            s3_key = await _upload_to_s3(url, interaction_id)
            await _db_write_recording_key(interaction_id, s3_key)
            log_event("recording_uploaded", interaction_id=interaction_id, attempt=attempt)
            return s3_key
        
        wait = min(5 * (2 ** attempt), 120)  # 5s, 10s, 20s, 40s, 80s, 120s, 120s, 120s
        log_event("recording_poll_retry", interaction_id=interaction_id, 
                  attempt=attempt, next_retry_in=wait)
        await asyncio.sleep(wait)
    
    # All attempts exhausted — this is a structured failure, not a silent skip
    log_event("recording_permanently_failed", interaction_id=interaction_id,
              level="ERROR", alert=True)
    await _db_write_recording_failed(interaction_id)
    return None
```

**Total wall time:** Up to ~9 minutes (5+10+20+40+80+120+120+120). This runs in parallel with LLM analysis, so it doesn't block the dashboard result.

**Visibility:** Every attempt is a structured log event with `interaction_id`. `recording_permanently_failed` triggers an alert (see §9). An ops engineer can find every recording attempt for interaction X with one log query.

**Idempotency:** If the recording task is retried (worker crash), `_fetch_exotel_recording_url` is safe to call multiple times. `_upload_to_s3` uses the `interaction_id` as the key — re-upload is idempotent.

---

## 8. Reliability & Durability

### The problem with Celery + Redis

A task in Redis broker memory is gone on restart. The current dual-retry approach (Celery + PostCallRetryQueue) creates duplicate processing risk.

### Solution: Postgres-backed processing events

New table `processing_events` (see §10). Before a task does any work, it writes a row. On completion, it updates the row. A reconciliation job runs every 5 minutes:

```sql
-- Find interactions that started processing but never completed
SELECT interaction_id FROM processing_events
WHERE status = 'STARTED'
AND started_at < NOW() - INTERVAL '10 minutes'
AND interaction_id NOT IN (
    SELECT interaction_id FROM processing_events WHERE status = 'COMPLETED'
);
```

These interactions are re-enqueued. Because all LLM writes are idempotent (we use `ON CONFLICT DO UPDATE`), re-processing is safe.

### What "no permanent loss" means operationally

1. Call ends → interaction row written immediately with `status=QUEUED`
2. Task enqueued → `processing_events` row written with `status=QUEUED`
3. Task starts → row updated to `STARTED`
4. Task completes → row updated to `COMPLETED`, `interaction.status = PROCESSED`
5. Reconciliation: any `STARTED` row older than 10 minutes is re-enqueued

At no point can a task silently disappear. The worst case is a duplicate run, which is handled by idempotent writes.

---

## 9. Auditability & Observability

### Structured log fields

Every log event includes:
```json
{
  "event": "recording_poll_retry",
  "interaction_id": "uuid",
  "customer_id": "uuid",
  "campaign_id": "uuid",
  "session_id": "uuid",
  "timestamp": "ISO8601",
  "attempt": 2,
  "next_retry_in": 20,
  "worker_id": "celery@hostname-123"
}
```

`interaction_id` is the correlation ID for tracing an interaction end-to-end across all log events.

### Debugging a failed interaction 3 days later

```sql
-- All processing events for the interaction
SELECT * FROM processing_events
WHERE interaction_id = 'uuid'
ORDER BY created_at;

-- All token budget usage attributed to this interaction
SELECT * FROM token_ledger
WHERE interaction_id = 'uuid';
```

In logs (e.g. CloudWatch Logs Insights):
```
fields @timestamp, event, attempt, error
| filter interaction_id = "uuid"
| sort @timestamp asc
```

### Alert conditions

| Alert | Condition | Severity |
|-------|-----------|----------|
| Recording permanently failed | `recording_permanently_failed` event | P2 |
| Task stuck in STARTED > 10m | Reconciliation job detects stale row | P2 |
| Customer budget exhausted | Customer queue depth > 1000 AND budget at 0 | P1 |
| Global TPM > 95% | `budget:global:used_tokens:{bucket}` > 95% of limit | P1 |
| HOT lane queue depth > 500 | Celery queue depth check | P1 |
| LLM 429 received | Should never happen with budget guard; if it does, P0 | P0 |

---

## 10. Data Model

```sql
-- New: per-customer LLM allocation config
CREATE TABLE customer_llm_config (
    customer_id UUID PRIMARY KEY REFERENCES customers(id),
    tokens_per_minute_reserved INTEGER NOT NULL DEFAULT 0,
    priority VARCHAR(20) NOT NULL DEFAULT 'standard',  -- high, standard
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- New: durable processing event log (the "what happened" table)
CREATE TABLE processing_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    interaction_id UUID NOT NULL REFERENCES interactions(id),
    customer_id UUID NOT NULL,
    campaign_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL,  -- classify_start, classify_complete, analysis_start, analysis_complete, recording_attempt, recording_complete, recording_failed, signal_jobs_complete
    status VARCHAR(20) NOT NULL,       -- QUEUED, STARTED, COMPLETED, FAILED, WAITING_BUDGET
    processing_lane VARCHAR(10),       -- hot, cold, skip
    attempt_number INTEGER DEFAULT 0,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_pe_interaction ON processing_events(interaction_id);
CREATE INDEX idx_pe_status ON processing_events(status, started_at) WHERE status = 'STARTED';
CREATE INDEX idx_pe_customer_created ON processing_events(customer_id, created_at);

-- New: token consumption ledger (source of truth for billing)
CREATE TABLE token_ledger (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id UUID NOT NULL,
    campaign_id UUID NOT NULL,
    interaction_id UUID NOT NULL REFERENCES interactions(id),
    processing_phase VARCHAR(20) NOT NULL,  -- classify, full_analysis
    estimated_tokens INTEGER NOT NULL,
    actual_tokens INTEGER,
    model VARCHAR(50) NOT NULL,
    minute_bucket BIGINT NOT NULL,  -- unix_timestamp // 60
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_tl_customer_bucket ON token_ledger(customer_id, minute_bucket);
CREATE INDEX idx_tl_interaction ON token_ledger(interaction_id);

-- Modify existing interactions table
ALTER TABLE interactions
    ADD COLUMN processing_lane VARCHAR(10),       -- hot, cold, skip, null=unclassified
    ADD COLUMN processing_status VARCHAR(20) DEFAULT 'PENDING',  -- PENDING, QUEUED, CLASSIFIED, ANALYSED, COMPLETED, FAILED
    ADD COLUMN recording_status VARCHAR(20) DEFAULT 'PENDING',   -- PENDING, UPLOADED, FAILED
    ADD COLUMN estimated_tokens INTEGER,
    ADD COLUMN actual_tokens INTEGER;

-- Remove: postcall_celery_task_id (replaced by processing_events)
-- Keep: retry_count, error_log for backwards compat but processing_events is the canonical record

-- New: analysis results table (separate from interaction_metadata JSONB blob)
CREATE TABLE interaction_analysis (
    interaction_id UUID PRIMARY KEY REFERENCES interactions(id),
    call_stage VARCHAR(100) NOT NULL,
    entities JSONB DEFAULT '{}',
    summary TEXT,
    raw_llm_response JSONB,
    tokens_used INTEGER NOT NULL,
    latency_ms FLOAT,
    model VARCHAR(50),
    analysed_at TIMESTAMPTZ DEFAULT NOW(),
    -- If reprocessed, we keep the latest and log the previous in processing_events
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Why a separate `interaction_analysis` table?** The current JSONB blob in `interactions.interaction_metadata` is the only record of analysis results. A re-run silently overwrites the previous result. A dedicated table with `updated_at` and the old value retained in `processing_events.metadata` gives us a history.

---

## 11. Security

### Sensitive data inventory

| Data | Sensitivity | Location |
|------|-------------|----------|
| Call transcripts | High — conversation PII | `interactions.conversation_data` (JSONB) |
| Call recordings | High — audio of conversation | S3 (`recordings/{interaction_id}.mp3`) |
| Lead PII (name, phone, email) | High | `leads` table |
| LLM API key | Critical | Env var / secrets manager |
| Analysis summaries | Medium — derived from transcripts | `interaction_analysis.summary` |

### Protection strategy

**At rest:**
- Postgres column-level encryption for `conversation_data` using `pgcrypto` (`pgp_sym_encrypt` with a key from AWS KMS or Vault). The symmetric key is rotated quarterly and the old key is kept for decryption of existing rows.
- S3 recordings: SSE-S3 minimum, SSE-KMS preferred. Bucket policy denies `s3:GetObject` except from the application role.
- `leads.phone` and `leads.email` are encrypted at the column level. The application decrypts on read; the DB stores ciphertext.

**In transit:**
- All LLM API calls over TLS 1.2+. No transcripts logged at INFO in production — they're logged only at DEBUG behind a feature flag.
- Internal service communication over mTLS (or VPC-only with security groups).

**Access control:**
- LLM API key in AWS Secrets Manager, rotated monthly. Workers fetch it at startup, not hardcoded.
- Recordings in S3 are accessed via pre-signed URLs with 1-hour expiry. No public bucket.
- `interaction_analysis.summary` is returned to the customer's dashboard; raw `entities` are shown only to users with appropriate permissions.

**Data retention:**
- Transcripts: 90 days then deleted (regulatory default). Configurable per customer via `customer_llm_config`.
- Recordings: 30 days then deleted. S3 lifecycle policy.
- Token ledger: retained 2 years for billing disputes.

---

## 12. API Interface

**No changes to the webhook endpoint signature** (`POST /session/{sid}/interaction/{iid}/end`). Reasoning: the telephony provider (Exotel) is the caller — we don't control that interface and changing it requires coordinated deployment with a third party.

**What changes internally:** The endpoint now writes the interaction row immediately (with `status=QUEUED`) and enqueues just the lightweight classification task, not the full processing task. The payload passed to Celery is the same — no external contract change.

**One addition:** The endpoint returns a `processing_lane` field in the response body once classification is done. If classification is async (for high-load cases), the response returns `processing_lane: null` and the dashboard polls `/interaction/{id}/status`.

---

## 13. Trade-offs & Alternatives Considered

| Option | Why Considered | Why Rejected / What Chosen |
|--------|---------------|---------------------------|
| Replace Celery with Temporal | True workflow durability, visible state machine | Significant infrastructure change; out of scope. Mitigated with Postgres event table. |
| Replace Redis broker with SQS | Durable messages, no broker restart risk | Adds AWS dependency. Mitigated with Postgres reconciliation. |
| Keyword-based lane classification | Zero token cost | Fails on Hinglish and multilingual transcripts as shown in sample data. Two-phase LLM wins. |
| One LLM call with priority flag | Simpler code | Doesn't solve timing problem — hot and cold calls still compete for same token budget. Separate queues give hot calls guaranteed first access. |
| Streaming LLM response | Faster time-to-first-token | call_stage is in the first 10 tokens of the response — could be a win. Deferred to "with more time" list. |
| Per-customer Celery queues | Complete isolation | O(K) queues at 100+ customers becomes a management burden. Two lanes (hot/cold) + token budget scheduler achieves equivalent fairness with 2 queues. |

---

## 14. Known Weaknesses

1. **Phase 1 classification still consumes tokens.** Even "skip" calls go through Phase 1 unless they're caught by the turn-count check. A pure keyword pre-filter before Phase 1 (for obvious cases like "wrong number") could reduce this further.

2. **Token estimation is rough.** We estimate `len(transcript_chars) // 4` tokens. Real GPT tokenisation is subword-based and this estimate can be off by 20–30% for Hinglish text. Reconciliation corrects it after the fact, but during a burst, over-estimation means we under-use our budget; under-estimation risks 429s.

3. **Reconciliation job has a 10-minute recovery window.** A stuck task won't be re-enqueued for up to 10 minutes. For HOT calls, this is acceptable but not ideal.

4. **No CRM push implemented.** Signal jobs are still fire-and-forget. A proper CRM push with retry and status tracking is in the "should implement" list but the implementation below only stubs it.

5. **Single LLM provider.** The system has one API key, one provider. If the provider has an outage, there's no fallback. Provider abstraction layer with a backup provider is a future improvement.

---

## 15. What I Would Do With More Time

1. **Provider abstraction + fallback.** Abstract the LLM call behind a provider interface. If the primary provider returns 429 or 5xx, route to a secondary (e.g., Anthropic as fallback for OpenAI). Token budget tracking would need to account for different pricing.

2. **Streaming call_stage extraction.** Stream the LLM response and extract `call_stage` from the first 10–15 tokens. Trigger signal jobs and lead stage update before the full response is complete. Reduces time-to-action for HOT calls from ~3.5s to ~0.5s.

3. **Dead-letter queue with replayability.** Any interaction that fails all retries goes to a DLQ table in Postgres. An admin UI shows the contents and allows selective replay with modified parameters (e.g., different model, increased token budget).

4. **Per-customer dashboard for budget visibility.** Real-time view of tokens used / remaining / queued per customer per minute. Lets customers self-serve and reduces support tickets.

5. **Gradual dialler backpressure.** Replace the binary circuit breaker with a signal from the post-call pipeline to the dialler: "your HOT queue depth is N, COLD queue is M, consider pacing dispatch at X calls/min." The dialler can decide; it's not frozen.

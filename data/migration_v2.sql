-- Migration: Post-call pipeline redesign
-- Adds: customer_llm_config, processing_events, token_ledger,
--       interaction_analysis, and columns to interactions

BEGIN;

-- ── 1. Per-customer LLM allocation ─────────────────────────────────────────

CREATE TABLE IF NOT EXISTS customer_llm_config (
    customer_id UUID PRIMARY KEY,
    tokens_per_minute_reserved INTEGER NOT NULL DEFAULT 0,
    priority VARCHAR(20) NOT NULL DEFAULT 'standard',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE customer_llm_config IS
    'Per-customer LLM token budget. tokens_per_minute_reserved is a guaranteed '
    'floor; customers may burst above it into unallocated headroom. '
    'Updated via API when customers change their plan.';

-- ── 2. Durable processing event log ────────────────────────────────────────

CREATE TABLE IF NOT EXISTS processing_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    interaction_id UUID NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    customer_id UUID NOT NULL,
    campaign_id UUID NOT NULL,

    -- Lifecycle events:
    --   classify_started, classify_completed, classify_failed
    --   budget_check_started, budget_check_completed, budget_check_waiting_budget
    --   llm_analysis_started, llm_analysis_completed, llm_analysis_failed
    --   recording_poll_attempt, recording_uploaded, recording_permanently_failed
    --   signal_jobs_started, signal_jobs_completed, signal_jobs_failed
    --   lead_stage_update_started, lead_stage_update_completed, lead_stage_update_failed
    event_type VARCHAR(80) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'INFO',  -- INFO, STARTED, COMPLETED, FAILED, WAITING_BUDGET

    processing_lane VARCHAR(10),   -- hot, cold, skip
    attempt_number INTEGER DEFAULT 0,
    error_message TEXT,

    -- Arbitrary metadata: estimated_tokens, actual_tokens, call_stage, s3_key, etc.
    metadata JSONB DEFAULT '{}',

    started_at TIMESTAMPTZ,    -- set when status = STARTED
    completed_at TIMESTAMPTZ,  -- set when status = COMPLETED or FAILED
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pe_interaction
    ON processing_events(interaction_id, created_at DESC);

-- Used by reconciliation job to find stuck tasks
CREATE INDEX IF NOT EXISTS idx_pe_stuck
    ON processing_events(status, started_at)
    WHERE status = 'STARTED';

CREATE INDEX IF NOT EXISTS idx_pe_customer_recent
    ON processing_events(customer_id, created_at DESC);

COMMENT ON TABLE processing_events IS
    'Append-only audit log of every processing event for every interaction. '
    'Used for debugging, alerting, and reconciliation of stuck tasks. '
    'Do not update rows — always insert new events.';

-- ── 3. Token consumption ledger (billing source of truth) ──────────────────

CREATE TABLE IF NOT EXISTS token_ledger (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id UUID NOT NULL,
    campaign_id UUID NOT NULL,
    interaction_id UUID NOT NULL REFERENCES interactions(id),

    -- 'classify' or 'full_analysis'
    processing_phase VARCHAR(20) NOT NULL,

    estimated_tokens INTEGER NOT NULL,  -- pre-call estimate (for budget gating)
    actual_tokens INTEGER,              -- post-call actual (from LLM usage field)

    model VARCHAR(100) NOT NULL,
    -- 60-second bucket: unix_timestamp // 60. Enables per-minute aggregation.
    minute_bucket BIGINT NOT NULL,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tl_customer_bucket
    ON token_ledger(customer_id, minute_bucket);
CREATE INDEX IF NOT EXISTS idx_tl_campaign
    ON token_ledger(campaign_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tl_interaction
    ON token_ledger(interaction_id);

COMMENT ON TABLE token_ledger IS
    'Append-only record of every token consumed. Estimated tokens are written '
    'before the LLM call (for budget gating); actual tokens are backfilled after. '
    'This table is the authoritative source for customer billing and usage reports.';

-- ── 4. Analysis results table (replaces JSONB blob in interactions) ─────────

CREATE TABLE IF NOT EXISTS interaction_analysis (
    interaction_id UUID PRIMARY KEY REFERENCES interactions(id) ON DELETE CASCADE,
    call_stage VARCHAR(100) NOT NULL DEFAULT 'unknown',
    entities JSONB DEFAULT '{}',
    summary TEXT,
    raw_llm_response JSONB,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    latency_ms FLOAT,
    model VARCHAR(100),
    analysed_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE interaction_analysis IS
    'Structured analysis result for each interaction. Separated from the '
    'interactions table to allow the dashboard to query results without '
    'touching the large conversation_data JSONB column. ON CONFLICT DO UPDATE '
    'makes reprocessing idempotent.';

-- ── 5. Extend interactions table ────────────────────────────────────────────

ALTER TABLE interactions
    ADD COLUMN IF NOT EXISTS processing_lane VARCHAR(10),
    ADD COLUMN IF NOT EXISTS processing_status VARCHAR(20) DEFAULT 'PENDING',
    ADD COLUMN IF NOT EXISTS recording_status VARCHAR(20) DEFAULT 'PENDING',
    ADD COLUMN IF NOT EXISTS estimated_tokens INTEGER,
    ADD COLUMN IF NOT EXISTS actual_tokens INTEGER;

CREATE INDEX IF NOT EXISTS idx_interactions_processing_status
    ON interactions(processing_status)
    WHERE processing_status NOT IN ('COMPLETED', 'SKIPPED');

COMMENT ON COLUMN interactions.processing_lane IS
    'Assigned by Phase 1 classification: hot, cold, or skip.';
COMMENT ON COLUMN interactions.processing_status IS
    'Pipeline status: PENDING → QUEUED → CLASSIFIED → ANALYSED → COMPLETED | FAILED | SKIPPED.';
COMMENT ON COLUMN interactions.recording_status IS
    'Recording pipeline status: PENDING → UPLOADED | FAILED.';

-- ── 6. Useful aggregation view for per-customer dashboards ──────────────────

CREATE OR REPLACE VIEW customer_token_usage_by_minute AS
SELECT
    customer_id,
    minute_bucket,
    minute_bucket * 60 AS bucket_start_epoch,
    to_timestamp(minute_bucket * 60) AS bucket_start_ts,
    SUM(estimated_tokens) AS estimated_tokens_total,
    SUM(actual_tokens) AS actual_tokens_total,
    COUNT(*) AS calls_processed
FROM token_ledger
GROUP BY customer_id, minute_bucket;

COMMENT ON VIEW customer_token_usage_by_minute IS
    'Aggregated token usage per customer per 60-second bucket. '
    'Useful for dashboards, billing exports, and rate limit monitoring.';

COMMIT;

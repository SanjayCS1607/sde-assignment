"""
Tests for the redesigned post-call processing pipeline.

Covers the acceptance criteria from the README:
  AC1 — System never fires LLM requests beyond configured rate limits
  AC2 — Per-customer token budget enforced
  AC4 — Recording poller retries with backoff; never silently skips
  AC5/AC6 — Audit events emitted for each stage
  AC7 — No binary dialler freeze
  AC8 — Short transcripts never consume LLM quota
"""

import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.token_budget import TokenBudgetScheduler
from src.services.lane_classifier import classify_lane, is_short_call
from src.services.recording import fetch_and_upload_recording, RECORDING_MAX_ATTEMPTS
from src.services.audit_logger import AuditLogger


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_redis():
    """In-memory Redis mock using a simple dict store."""
    store: dict = {}

    async def get(key):
        return store.get(key)

    async def set(key, value, ex=None):
        store[key] = str(value)

    async def incrby(key, amount):
        current = int(store.get(key, 0))
        store[key] = str(current + amount)
        return current + amount

    async def decrby(key, amount):
        current = int(store.get(key, 0))
        new_val = max(0, current - amount)
        store[key] = str(new_val)
        return new_val

    async def expireat(key, when):
        pass  # no-op in tests

    async def sadd(key, *values):
        if key not in store:
            store[key] = set()
        store[key].add(values[0])

    class FakePipeline:
        def __init__(self):
            self._commands = []

        def get(self, key):
            self._commands.append(("get", key))
            return self

        def incrby(self, key, amount):
            self._commands.append(("incrby", key, amount))
            return self

        def decrby(self, key, amount):
            self._commands.append(("decrby", key, amount))
            return self

        def expireat(self, key, when):
            return self

        def sadd(self, key, *values):
            return self

        async def execute(self):
            results = []
            for cmd in self._commands:
                if cmd[0] == "get":
                    results.append(store.get(cmd[1]))
                elif cmd[0] == "incrby":
                    current = int(store.get(cmd[1], 0))
                    store[cmd[1]] = str(current + cmd[2])
                    results.append(store[cmd[1]])
                elif cmd[0] == "decrby":
                    current = int(store.get(cmd[1], 0))
                    store[cmd[1]] = str(max(0, current - cmd[2]))
                    results.append(store[cmd[1]])
                else:
                    results.append(None)
            return results

    mock = MagicMock()
    mock.get = get
    mock.set = set
    mock.incrby = incrby
    mock.decrby = decrby
    mock.expireat = expireat
    mock.sadd = sadd
    mock.pipeline = lambda: FakePipeline()
    mock._store = store

    return mock


@pytest.fixture
def scheduler(mock_redis):
    sched = TokenBudgetScheduler()
    return sched, mock_redis


# ── AC1: Rate limit enforcement ───────────────────────────────────────────────

class TestRateLimitEnforcement:
    """AC1: System never fires LLM requests beyond configured rate limits."""

    @pytest.mark.asyncio
    async def test_budget_exhausted_returns_none(self, scheduler, mock_redis):
        """When the global budget is fully used, acquire_budget returns None."""
        sched, redis = scheduler

        bucket = int(time.time()) // 60
        # Pre-fill global budget to capacity
        redis._store[f"budget:global:{bucket}"] = "90000"  # at limit
        redis._store[f"budget:customer1:{bucket}"] = "0"
        redis._store["budget:customer1:reserved"] = "20000"

        with patch("src.services.token_budget.redis_client", redis):
            result = await sched.acquire_budget("customer1", 1500)

        assert result is None, "Should return None when global budget exhausted"

    @pytest.mark.asyncio
    async def test_budget_available_returns_reservation(self, scheduler, mock_redis):
        """When budget is available, acquire_budget returns a reservation dict."""
        sched, redis = scheduler

        redis._store["budget:customer1:reserved"] = "20000"

        with patch("src.services.token_budget.redis_client", redis), \
             patch("src.services.token_budget.settings") as mock_settings:
            mock_settings.LLM_TOKENS_PER_MINUTE = 90000
            result = await sched.acquire_budget("customer1", 1500)

        assert result is not None
        assert result["customer_id"] == "customer1"
        assert result["estimated_tokens"] == 1500

    @pytest.mark.asyncio
    async def test_burst_of_1000_calls_stays_within_budget(self, scheduler, mock_redis):
        """
        AC1: Simulate burst of 1000 concurrent budget acquisitions.
        Total tokens requested: 1000 × 1500 = 1,500,000.
        Budget: 90,000 tokens/min.
        Expected: at most 60 calls succeed (90,000 / 1500 = 60).
        No 429 should be triggered — budget denials return None, not exceptions.
        """
        sched, redis = scheduler
        redis._store["budget:customer1:reserved"] = "90000"

        approved = 0
        denied = 0

        with patch("src.services.token_budget.redis_client", redis), \
             patch("src.services.token_budget.settings") as mock_settings:
            mock_settings.LLM_TOKENS_PER_MINUTE = 90000

            for _ in range(1000):
                result = await sched.acquire_budget("customer1", 1500)
                if result is not None:
                    approved += 1
                else:
                    denied += 1

        # At 1500 tokens each, max 60 should be approved within 90K budget
        assert approved <= 60, f"Too many approved: {approved} (budget would be exceeded)"
        assert denied >= 940, f"Too few denied: {denied}"
        # No exception was raised — no 429 surfaced to callers
        print(f"Approved: {approved}, Denied: {denied}")


# ── AC2: Per-customer budget isolation ────────────────────────────────────────

class TestPerCustomerBudget:
    """AC2: Customer A's budget does not consume Customer B's allocation."""

    @pytest.mark.asyncio
    async def test_customer_a_exhaustion_does_not_block_customer_b(
        self, scheduler, mock_redis
    ):
        """
        Exhaust Customer A's reserved budget.
        Customer B's calls should still be approved up to B's limit.
        """
        sched, redis = scheduler
        bucket = int(time.time()) // 60

        # Customer A: 20,000 reserved, already fully used
        redis._store["budget:customerA:reserved"] = "20000"
        redis._store[f"budget:customerA:{bucket}"] = "20000"  # exhausted

        # Customer B: 15,000 reserved, nothing used yet
        redis._store["budget:customerB:reserved"] = "15000"
        redis._store[f"budget:customerB:{bucket}"] = "0"

        with patch("src.services.token_budget.redis_client", redis), \
             patch("src.services.token_budget.settings") as mock_settings:
            mock_settings.LLM_TOKENS_PER_MINUTE = 90000

            result_a = await sched.acquire_budget("customerA", 1500)
            result_b = await sched.acquire_budget("customerB", 1500)

        assert result_a is None, "Customer A should be budget-denied"
        assert result_b is not None, "Customer B should be approved (own budget available)"

    @pytest.mark.asyncio
    async def test_customer_reservation_respected(self, scheduler, mock_redis):
        """
        Customer with a reservation gets their guaranteed allocation.
        """
        sched, redis = scheduler
        redis._store["budget:vip_customer:reserved"] = "30000"

        with patch("src.services.token_budget.redis_client", redis), \
             patch("src.services.token_budget.settings") as mock_settings:
            mock_settings.LLM_TOKENS_PER_MINUTE = 90000

            # VIP customer should be able to approve up to 30,000 tokens
            approved = 0
            for _ in range(25):  # 25 × 1200 = 30,000 tokens
                r = await sched.acquire_budget("vip_customer", 1200)
                if r:
                    approved += 1

        assert approved == 25, f"VIP customer reserved 30K, should get 25 slots of 1200 tokens"


# ── AC4: Recording poller ─────────────────────────────────────────────────────

class TestRecordingPoller:
    """AC4: Recording poller retries with backoff; never silently skips."""

    @pytest.mark.asyncio
    async def test_recording_eventually_available(self):
        """Recording becomes available on attempt 3 — should succeed."""
        attempts = []

        async def mock_fetch(call_sid, account_id):
            attempts.append(len(attempts) + 1)
            if len(attempts) >= 3:
                return "https://exotel.com/recording.mp3"
            return None

        async def mock_upload(url, interaction_id):
            return f"recordings/{interaction_id}.mp3"

        with patch("src.services.recording._fetch_exotel_recording_url", mock_fetch), \
             patch("src.services.recording._upload_to_s3", mock_upload), \
             patch("src.services.recording.asyncio.sleep", AsyncMock()):

            result = await fetch_and_upload_recording(
                interaction_id="test-interaction",
                call_sid="call-123",
                exotel_account_id="account-123",
            )

        assert result == "recordings/test-interaction.mp3"
        assert len(attempts) == 3, f"Expected 3 attempts, got {len(attempts)}"

    @pytest.mark.asyncio
    async def test_recording_permanently_failed_logs_error(self, caplog):
        """When all attempts exhausted, logs ERROR with alert=True (not silent)."""
        async def mock_fetch(call_sid, account_id):
            return None  # Always not available

        with patch("src.services.recording._fetch_exotel_recording_url", mock_fetch), \
             patch("src.services.recording.asyncio.sleep", AsyncMock()), \
             caplog.at_level(logging.ERROR):

            result = await fetch_and_upload_recording(
                interaction_id="test-interaction-fail",
                call_sid="call-fail",
                exotel_account_id="account-123",
            )

        assert result is None
        # Should have logged an ERROR-level event (not just silently returned None)
        error_logs = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_logs) > 0, "Expected ERROR log on permanent failure"
        assert any("permanently_failed" in r.message or "recording" in r.message.lower()
                   for r in error_logs), "ERROR log should mention recording failure"

    @pytest.mark.asyncio
    async def test_recording_retries_max_attempts_times(self):
        """Polling should attempt exactly RECORDING_MAX_ATTEMPTS times."""
        attempts = []

        async def mock_fetch(call_sid, account_id):
            attempts.append(1)
            return None

        with patch("src.services.recording._fetch_exotel_recording_url", mock_fetch), \
             patch("src.services.recording.asyncio.sleep", AsyncMock()):

            await fetch_and_upload_recording("iid", "csid", "acid")

        assert len(attempts) == RECORDING_MAX_ATTEMPTS

    @pytest.mark.asyncio
    async def test_recording_network_error_retries(self):
        """Network errors during polling are retried, not swallowed."""
        call_count = [0]

        async def mock_fetch(call_sid, account_id):
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Connection timeout")
            return "https://example.com/recording.mp3"

        async def mock_upload(url, iid):
            return f"recordings/{iid}.mp3"

        with patch("src.services.recording._fetch_exotel_recording_url", mock_fetch), \
             patch("src.services.recording._upload_to_s3", mock_upload), \
             patch("src.services.recording.asyncio.sleep", AsyncMock()):

            result = await fetch_and_upload_recording("iid", "csid", "acid")

        assert result == "recordings/iid.mp3"
        assert call_count[0] == 3


# ── AC8: Short transcripts skip LLM ───────────────────────────────────────────

class TestShortCallSkip:
    """AC8: Short transcripts (<4 turns) never consume LLM quota."""

    @pytest.mark.asyncio
    async def test_short_call_classified_as_skip(self):
        """<4 turns → skip lane, no LLM call."""
        conversation_data = {
            "transcript": [
                {"role": "agent", "content": "Hello—"},
                {"role": "customer", "content": "Wrong number."},
            ]
        }

        lane = await classify_lane(
            transcript_text="Hello— Wrong number.",
            conversation_data=conversation_data,
            interaction_id="test-short",
        )

        assert lane == "skip"

    def test_is_short_call_with_2_turns(self):
        data = {"transcript": [{"role": "agent", "content": "Hi"}, {"role": "customer", "content": "Bye"}]}
        assert is_short_call(data) is True

    def test_is_short_call_with_4_turns(self):
        data = {"transcript": [{"role": "agent", "content": f"msg{i}"} for i in range(4)]}
        assert is_short_call(data) is False

    @pytest.mark.asyncio
    async def test_short_call_does_not_call_llm(self):
        """Verify that classify_lane with a short transcript returns 'skip' without an LLM call."""
        conversation_data = {
            "transcript": [
                {"role": "agent", "content": "Hello"},
                {"role": "customer", "content": "No"},
            ]
        }

        llm_called = []

        async def mock_llm_classify(transcript, interaction_id):
            llm_called.append(True)
            return "unknown"

        with patch("src.services.lane_classifier._llm_classify", mock_llm_classify):
            lane = await classify_lane("Hello No", conversation_data, "test")

        assert lane == "skip"
        assert len(llm_called) == 0, "LLM should not be called for short transcripts"


# ── AC5/AC6: Audit trail ──────────────────────────────────────────────────────

class TestAuditLogging:
    """AC5/AC6: Every interaction has a complete audit trail; every failure includes interaction_id."""

    @pytest.mark.asyncio
    async def test_audit_logger_emits_started_and_completed(self, caplog):
        """AuditLogger emits STARTED and COMPLETED events with interaction_id."""
        with caplog.at_level(logging.INFO):
            async with AuditLogger("iid-123", "cust-1", "camp-1") as audit:
                await audit.started("llm_analysis", metadata={"estimated_tokens": 1500})
                await audit.completed("llm_analysis", metadata={"actual_tokens": 1423})

        messages = [r.message for r in caplog.records]
        assert any("started" in m.lower() or "llm_analysis" in m.lower() for m in messages)
        assert any("completed" in m.lower() or "llm_analysis" in m.lower() for m in messages)

    @pytest.mark.asyncio
    async def test_audit_logger_emits_failed_on_exception(self, caplog):
        """AuditLogger emits FAILED event when context exits with exception."""
        with caplog.at_level(logging.ERROR):
            try:
                async with AuditLogger("iid-456", "cust-1", "camp-1") as audit:
                    await audit.started("llm_analysis")
                    raise ValueError("LLM provider returned 500")
            except ValueError:
                pass

        # The context manager should have emitted a FAILED event
        error_logs = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_logs) > 0, "Should emit ERROR log on exception"

    @pytest.mark.asyncio
    async def test_audit_event_includes_interaction_id(self, caplog):
        """Every audit event carries the interaction_id (AC6)."""
        interaction_id = "trace-me-12345"

        with caplog.at_level(logging.INFO):
            async with AuditLogger(interaction_id, "cust-1", "camp-1") as audit:
                await audit.event("test_event", status="INFO", metadata={"key": "val"})

        # Find log records that mention this interaction
        matching = [
            r for r in caplog.records
            if interaction_id in str(getattr(r, "__dict__", {}))
               or interaction_id in r.getMessage()
        ]
        assert len(matching) > 0, f"No log record found with interaction_id={interaction_id}"


# ── AC7: No binary dialler freeze ─────────────────────────────────────────────

class TestNoBinaryFreeze:
    """
    AC7: Dialler is not binary-frozen when LLM is under load.

    The old circuit breaker froze the dialler for 1800s at 90% capacity.
    The new design uses budget acquisition failure → task re-queues itself.
    The dialler is never touched.
    """

    def test_no_circuit_breaker_import_in_tasks(self):
        """
        The new celery_tasks.py should not import PostCallCircuitBreaker.
        The dialler freeze lived in the circuit breaker — removing this import
        confirms the binary freeze is gone.
        """
        import inspect
        from src.tasks import celery_tasks
        source = inspect.getsource(celery_tasks)
        assert "PostCallCircuitBreaker" not in source, \
            "celery_tasks should not use PostCallCircuitBreaker (binary freeze)"
        assert "CIRCUIT_BREAKER_FREEZE" not in source, \
            "No hardcoded freeze duration should appear in task code"

    @pytest.mark.asyncio
    async def test_budget_exhaustion_does_not_raise_to_dialler(
        self, scheduler, mock_redis
    ):
        """Budget exhaustion returns None — no exception propagates to dialler."""
        sched, redis = scheduler
        bucket = int(time.time()) // 60
        redis._store[f"budget:global:{bucket}"] = "90000"  # fully exhausted

        with patch("src.services.token_budget.redis_client", redis), \
             patch("src.services.token_budget.settings") as mock_settings:
            mock_settings.LLM_TOKENS_PER_MINUTE = 90000

            # Should return None, not raise an exception
            result = await sched.acquire_budget("any_customer", 1500)

        assert result is None  # caller (Celery task) handles this by re-queuing itself


# ── Lane classification ────────────────────────────────────────────────────────

class TestLaneClassification:
    """Verify that sample transcripts route to the expected lanes."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("transcript,expected_lane", [
        # HOT
        ("Agent: Hello, am I speaking with Mr. Sharma? Customer: Haan ji. "
         "Agent: Our executive will visit tomorrow at 3:30 PM. Customer: Confirmed.", "hot"),
        # HOT — escalation
        ("Customer: I want to speak to a manager. Agent: I'll escalate immediately.", "hot"),
        # COLD
        ("Agent: Hello. Customer: Not interested. Please don't call again.", "cold"),
        # COLD — callback
        ("Agent: Can I call you later? Customer: Baad mein call karo. Agent: Sure.", "cold"),
    ])
    async def test_lane_classification(self, transcript, expected_lane):
        conversation_data = {
            "transcript": [
                {"role": "agent" if i % 2 == 0 else "customer", "content": t}
                for i, t in enumerate(transcript.split(". "))
                if len(t) > 5
            ]
        }
        # Ensure not short call
        while len(conversation_data["transcript"]) < 4:
            conversation_data["transcript"].append({"role": "agent", "content": "Thank you"})

        lane = await classify_lane(transcript, conversation_data, "test-iid")
        assert lane == expected_lane, f"Expected {expected_lane}, got {lane} for: {transcript[:60]}"

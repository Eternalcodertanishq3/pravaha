"""Tests for Phase 3 Reliability & Observability features."""

import json
import logging
import time
import pytest
from pravaha.observability.structured_logger import JSONFormatter, request_id_ctx
from pravaha.engine.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, CircuitState
from pravaha.observability.audit_trail import AuditTrail


def test_structured_logger_json_format():
    """Verify that JSONFormatter includes correlation ID and JSON structure."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=42,
        msg="Test log message",
        args=(),
        exc_info=None,
    )

    token = request_id_ctx.set("req-12345")
    try:
        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["message"] == "Test log message"
        assert data["request_id"] == "req-12345"
        assert "timestamp" in data
    finally:
        request_id_ctx.reset(token)


def test_circuit_breaker_transitions():
    """Verify CircuitBreaker state transitions: CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""
    cb = CircuitBreaker(
        name="test_cb",
        failure_threshold=2,
        recovery_timeout_seconds=0.1,  # Fast recovery for test
        half_open_success_threshold=1,
    )

    assert cb.state == CircuitState.CLOSED

    # Failure 1
    cb.record_failure(ValueError("err1"))
    assert cb.state == CircuitState.CLOSED

    # Failure 2 -> Threshold reached -> OPEN
    cb.record_failure(ValueError("err2"))
    assert cb.state == CircuitState.OPEN

    # Call while OPEN should raise CircuitBreakerOpenError
    with pytest.raises(CircuitBreakerOpenError):
        cb.call(lambda: "should fail")

    # Wait for recovery timeout
    time.sleep(0.15)
    assert cb.state == CircuitState.HALF_OPEN

    # Record success while HALF_OPEN -> CLOSED
    cb.record_success()
    assert cb.state == CircuitState.CLOSED


def test_audit_trail_hash_chain_integrity(tmp_path):
    """Verify AuditTrail hash chaining and tamper detection."""
    audit_file = tmp_path / "audit.jsonl"
    trail = AuditTrail(storage_path=str(audit_file))

    # Log 3 events
    rec0 = trail.log_event("AUTH_SUCCESS", "user-1", {"ip": "127.0.0.1"})
    rec1 = trail.log_event("POLICY_CHECK", "agent-2", {"allowed": True})
    rec2 = trail.log_event("TOOL_EXEC", "agent-2", {"cmd": "ls"})

    # Verify initial integrity
    is_valid, msg = trail.verify_integrity()
    assert is_valid is True
    assert "3 records intact" in msg

    # Simulate tampering: edit rec1 in the JSONL file
    with open(audit_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    tampered_data = json.loads(lines[1])
    tampered_data["details"]["allowed"] = False  # Alter payload
    lines[1] = json.dumps(tampered_data) + "\n"

    with open(audit_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    # Re-verify integrity — should fail tamper check!
    is_valid, msg = trail.verify_integrity()
    assert is_valid is False
    assert "tampered" in msg or "broken" in msg

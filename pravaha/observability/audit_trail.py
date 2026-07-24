"""Tamper-Resistant Audit Trail — Immutable, hash-chained log for security and agent events.

Uses cryptographic SHA-256 hash chaining (blockchain-style ledger):
Each record contains `prev_hash` pointing to the previous entry's SHA-256 digest.
Any modification, insertion, or deletion breaks the hash chain and is detected by verify_integrity().
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AuditRecord:
    """Single immutable audit log entry."""

    index: int
    timestamp: str
    event_type: str
    actor_id: str
    details: dict[str, Any]
    prev_hash: str
    record_hash: str


class AuditTrail:
    """Tamper-resistant append-only audit trail with SHA-256 hash chaining."""

    GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"

    def __init__(self, storage_path: str = "audit_trail.jsonl") -> None:
        self.storage_path = Path(storage_path)
        self._lock = threading.Lock()
        self._last_hash = self.GENESIS_HASH
        self._last_index = -1

        # Ensure directory exists and load state
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _compute_hash(
        self,
        index: int,
        timestamp: str,
        event_type: str,
        actor_id: str,
        details_str: str,
        prev_hash: str,
    ) -> str:
        """Compute SHA-256 digest over entry fields."""
        content = f"{index}:{timestamp}:{event_type}:{actor_id}:{details_str}:{prev_hash}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _load_state(self) -> None:
        """Load last record index and hash from existing file."""
        if not self.storage_path.exists() or self.storage_path.stat().st_size == 0:
            return

        try:
            with open(self.storage_path, encoding="utf-8") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:
                        data = json.loads(last_line)
                        self._last_index = data["index"]
                        self._last_hash = data["record_hash"]
        except Exception as e:
            logger.error(f"AuditTrail state load error: {e}")

    def log_event(self, event_type: str, actor_id: str, details: dict[str, Any] | None = None) -> AuditRecord:
        """Append a new tamper-evident audit record to the log.

        Args:
            event_type: Category of event (e.g., 'AGENT_ACTION', 'POLICY_CHANGE', 'AUTH_FAILURE').
            actor_id: ID of agent, user, or system component emitting event.
            details: Contextual payload dictionary.

        Returns:
            The created AuditRecord.
        """
        with self._lock:
            index = self._last_index + 1
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            details_clean = details or {}
            details_str = json.dumps(details_clean, sort_keys=True)
            prev_hash = self._last_hash

            record_hash = self._compute_hash(
                index, timestamp, event_type, actor_id, details_str, prev_hash
            )

            record = AuditRecord(
                index=index,
                timestamp=timestamp,
                event_type=event_type,
                actor_id=actor_id,
                details=details_clean,
                prev_hash=prev_hash,
                record_hash=record_hash,
            )

            # Write record as JSON line
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record)) + "\n")

            self._last_index = index
            self._last_hash = record_hash

            return record

    def verify_integrity(self) -> tuple[bool, str]:
        """Verify hash chain integrity across all historical records.

        Returns:
            Tuple of (is_valid: bool, status_message: str).
        """
        with self._lock:
            if not self.storage_path.exists():
                return True, "No audit trail file exists yet."

            expected_prev_hash = self.GENESIS_HASH
            expected_index = 0

            try:
                with open(self.storage_path, encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)

                        index = data["index"]
                        timestamp = data["timestamp"]
                        event_type = data["event_type"]
                        actor_id = data["actor_id"]
                        details = data["details"]
                        prev_hash = data["prev_hash"]
                        stored_hash = data["record_hash"]

                        # Check index order
                        if index != expected_index:
                            return False, f"Index mismatch at line {line_num}: expected {expected_index}, got {index}"

                        # Check previous hash link
                        if prev_hash != expected_prev_hash:
                            return False, f"Hash chain broken at line {line_num} (index {index}): prev_hash mismatch"

                        # Recompute current hash
                        details_str = json.dumps(details, sort_keys=True)
                        computed_hash = self._compute_hash(
                            index, timestamp, event_type, actor_id, details_str, prev_hash
                        )

                        if computed_hash != stored_hash:
                            return False, f"Record tampered at line {line_num} (index {index}): stored hash {stored_hash[:8]}... != computed {computed_hash[:8]}..."

                        expected_prev_hash = stored_hash
                        expected_index += 1

                return True, f"Audit trail verified: {expected_index} records intact."
            except Exception as e:
                return False, f"Verification failed with error: {e}"

"""Tests for Auditor Consensus module."""

import pytest
from typing import Any
from dataclasses import dataclass
from pravaha.swarm.consensus import AuditorConsensus, ConsensusResult

@dataclass
class MockOutput:
    role: str
    issues: list[dict[str, Any]]

def test_empty_inputs():
    consensus = AuditorConsensus()
    result = consensus.evaluate([])
    assert result.issues == []
    assert result.final_score == 0.0
    assert result.agreement_ratio == 0.0
    assert result.critical_count == 0

def test_single_auditor():
    consensus = AuditorConsensus()
    out = MockOutput(
        role="static",
        issues=[{"type": "lint", "severity": "MEDIUM", "description": "Line too long"}]
    )
    result = consensus.evaluate([out])
    assert len(result.issues) == 1
    assert result.issues[0]["severity"] == "MEDIUM"
    assert result.critical_count == 0
    # Score: MEDIUM = 4. weight = 1.0. Total possible = 10. actual = 4. final = 40.0
    assert result.final_score == 40.0
    assert result.agreement_ratio == 1.0

def test_deduplication():
    consensus = AuditorConsensus()
    out1 = MockOutput(
        role="static",
        issues=[{"type": "lint", "severity": "MEDIUM", "description": "Line is extremely long and exceeds limit"}]
    )
    out2 = MockOutput(
        role="llm",
        issues=[{"type": "style", "severity": "LOW", "description": "Line is extremely long and exceeds limit"}]
    )
    result = consensus.evaluate([out1, out2])
    assert len(result.issues) == 1
    # Should keep highest severity
    assert result.issues[0]["severity"] == "MEDIUM"
    assert result.agreement_ratio == 1.0

def test_weighted_scoring():
    consensus = AuditorConsensus()
    out1 = MockOutput(
        role="security",
        issues=[{"type": "vuln", "severity": "HIGH", "description": "SQL Injection found"}]
    )
    out2 = MockOutput(
        role="static",
        issues=[{"type": "vuln", "severity": "MEDIUM", "description": "SQL Injection found in module"}]
    )
    result = consensus.evaluate([out1, out2])
    assert len(result.issues) == 1
    assert result.issues[0]["severity"] == "HIGH"
    
    # Weight: security=1.5, static=1.0
    # Score possible = 10*1.5 + 10*1.0 = 25.0
    # Actual = 7*1.5 + 4*1.0 = 10.5 + 4.0 = 14.5
    # Final = (14.5 / 25.0) * 100 = 58.0
    assert result.final_score == 58.0

def test_critical_promotion():
    consensus = AuditorConsensus()
    out1 = MockOutput(role="security", issues=[{"type": "vuln", "severity": "CRITICAL", "description": "RCE"}])
    out2 = MockOutput(role="llm", issues=[{"type": "vuln", "severity": "CRITICAL", "description": "RCE vulnerability"}])
    result = consensus.evaluate([out1, out2])
    assert len(result.issues) == 1
    assert result.issues[0]["severity"] == "CRITICAL"
    assert result.critical_count == 1

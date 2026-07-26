"""Auditor Consensus & Weighted Voting Engine."""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConsensusResult:
    """Result of the auditor consensus evaluation."""

    issues: list[dict[str, Any]]
    final_score: float
    agreement_ratio: float
    critical_count: int


class AuditorConsensus:
    """Evaluates auditor outputs and computes a consensus score."""

    def __init__(self) -> None:
        self.weights = {
            "security": 1.5,
            "static": 1.0,
            "llm": 0.8,
        }
        self.severity_scores = {
            "CRITICAL": 10,
            "HIGH": 7,
            "MEDIUM": 4,
            "LOW": 1,
            "INFO": 0,
        }

    def _get_weight(self, agent_type: str) -> float:
        for key, weight in self.weights.items():
            if key in agent_type.lower():
                return weight
        return 1.0  # default weight

    def evaluate(self, outputs: list[Any]) -> ConsensusResult:
        """Evaluate consensus from multiple auditor outputs."""
        if not outputs:
            return ConsensusResult([], 0.0, 0.0, 0)

        deduped_issues: list[dict[str, Any]] = []
        issue_counts: dict[int, int] = {}
        critical_counts: dict[int, int] = {}
        total_possible_score = 0.0
        actual_score = 0.0

        for output in outputs:
            agent_type = getattr(output, "agent_role", getattr(output, "role", "unknown"))
            weight = self._get_weight(agent_type)
            issues = getattr(output, "issues", [])

            for issue in issues:
                desc = issue.get("description", "")
                severity = issue.get("severity", "INFO").upper()

                # Match against existing issues
                matched_idx = -1
                for idx, existing in enumerate(deduped_issues):
                    existing_desc = existing.get("description", "")
                    if desc and existing_desc:
                        d1, d2 = desc.lower().strip(), existing_desc.lower().strip()
                        if d1 == d2 or d1 in d2 or d2 in d1 or difflib.SequenceMatcher(None, d1, d2).ratio() >= 0.5:
                            matched_idx = idx
                            break

                score_val = self.severity_scores.get(severity, 0) * weight
                total_possible_score += 10 * weight  # assuming 10 is max score per issue
                actual_score += score_val

                if matched_idx >= 0:
                    issue_counts[matched_idx] += 1
                    if severity == "CRITICAL":
                        critical_counts[matched_idx] = critical_counts.get(matched_idx, 0) + 1

                    # Update severity if needed (keep highest)
                    existing_sev = deduped_issues[matched_idx].get("severity", "INFO").upper()
                    if self.severity_scores.get(severity, 0) > self.severity_scores.get(existing_sev, 0):
                        deduped_issues[matched_idx]["severity"] = severity
                else:
                    deduped_issues.append({
                        "type": issue.get("type", "unknown"),
                        "severity": severity,
                        "description": desc,
                    })
                    idx = len(deduped_issues) - 1
                    issue_counts[idx] = 1
                    if severity == "CRITICAL":
                        critical_counts[idx] = 1

        # Promote to CRITICAL if >= 2 auditors flag as CRITICAL
        for idx in critical_counts:
            if critical_counts[idx] >= 2:
                deduped_issues[idx]["severity"] = "CRITICAL"

        critical_count = sum(1 for issue in deduped_issues if issue.get("severity", "INFO").upper() == "CRITICAL")

        # Calculate agreement ratio (average number of auditors agreeing on an issue divided by total auditors)
        if not deduped_issues:
            agreement_ratio = 1.0
        else:
            avg_agreement = sum(issue_counts.values()) / len(deduped_issues)
            agreement_ratio = min(1.0, avg_agreement / len(outputs))

        final_score = (actual_score / total_possible_score * 100) if total_possible_score > 0 else 0.0

        return ConsensusResult(
            issues=deduped_issues,
            final_score=round(final_score, 2),
            agreement_ratio=round(agreement_ratio, 2),
            critical_count=critical_count,
        )

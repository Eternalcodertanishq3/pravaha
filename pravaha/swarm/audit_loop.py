"""Audit Loop — The self-healing feedback cycle.

After worker agents produce output, audit agents scan it.
If issues are found, PatchApplier fixes them and the output
is re-audited. Maximum 3 iterations. If issues remain after
3 loops, output is returned with an audit_report attached.

Phase 5: Core self-healing innovation — Pravaha's identity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pravaha.swarm.agents.base_agent import SharedContext

logger = logging.getLogger(__name__)


@dataclass
class AuditResult:
    """Structured result from the complete audit pipeline."""

    final_output: str
    issues_found: list[dict[str, Any]] = field(default_factory=list)
    patches_applied: list[str] = field(default_factory=list)
    iterations: int = 0
    satisfaction_score: float = 0.0
    audit_report: str = ""
    passed: bool = False


class AuditLoop:
    """Run audit agents in a feedback loop until quality threshold is met.

    The audit pipeline runs in this order:
    1. syntax_audit     — structural validity
    2. type_safety      — type correctness
    3. security_audit   — vulnerability scan
    4. logic_flaw       — reasoning check
    5. consistency_guard — cross-agent consistency
    6. hallucination_hunter — factual accuracy
    7. edge_case_hunter — missing edge cases
    8. performance_profiler — performance issues
    9. output_verifier  — task satisfaction score
    """

    AUDIT_PIPELINE = [
        "syntax_audit",
        "type_safety",
        "security_audit",
        "logic_flaw",
        "consistency_guard",
        "hallucination_hunter",
        "edge_case_hunter",
        "performance_profiler",
        "output_verifier",
    ]

    def __init__(
        self,
        max_iterations: int = 3,
        min_score: float = 70.0,
    ) -> None:
        self.max_iterations = max_iterations
        self.min_score = min_score

    async def run(
        self,
        output: str,
        task: str,
        context: SharedContext,
        orchestrator: Any,
        engine: Any,
        output_type: str = "code",
    ) -> AuditResult:
        """Run the full audit pipeline with patching loop.

        Args:
            output: The worker output to audit.
            task: The original task description.
            context: Shared context with all agent outputs.
            orchestrator: SwarmOrchestrator for agent execution.
            engine: AsyncPravahaEngine for generation.
            output_type: One of 'code', 'text', 'analysis'.

        Returns:
            AuditResult with final output, issues, and score.
        """
        current_output = output
        all_issues: list[dict[str, Any]] = []
        all_patches: list[str] = []
        score = 0.0

        # Select relevant auditors based on output type
        pipeline = self._select_auditors(output_type)

        for iteration in range(self.max_iterations):
            context.output = current_output
            context.code = current_output if output_type == "code" else context.code
            iteration_issues: list[dict[str, Any]] = []

            # Run each auditor
            for auditor_name in pipeline:
                result = await orchestrator.execute_agent(auditor_name, task, engine)
                iteration_issues.extend(result.issues)

            if not iteration_issues:
                # Get final score
                verifier_result = context.agent_outputs.get("output_verifier")
                score = verifier_result.metadata.get("score", 80) if verifier_result else 80.0
                logger.info(f"Audit PASSED: iteration {iteration + 1}, score {score}")
                return AuditResult(
                    final_output=current_output,
                    issues_found=all_issues,
                    patches_applied=all_patches,
                    iterations=iteration + 1,
                    satisfaction_score=score,
                    passed=True,
                )

            # Issues found — apply patches
            all_issues.extend(iteration_issues)
            context.audit_reports.append(
                {
                    "iteration": iteration + 1,
                    "issues": iteration_issues,
                }
            )

            patch_result = await orchestrator.execute_agent("patch_applier", task, engine)
            current_output = patch_result.output
            all_patches.extend(patch_result.patches)
            logger.info(
                f"Audit iteration {iteration + 1}: {len(iteration_issues)} issues, "
                f"applied {len(patch_result.patches)} patches"
            )

        # Max iterations reached
        verifier_result = context.agent_outputs.get("output_verifier")
        score = verifier_result.metadata.get("score", 50) if verifier_result else 50.0

        report = (
            f"Audit completed after {self.max_iterations} iterations. "
            f"Score: {score}/100. Issues remaining: {len(all_issues)}"
        )
        return AuditResult(
            final_output=current_output,
            issues_found=all_issues,
            patches_applied=all_patches,
            iterations=self.max_iterations,
            satisfaction_score=score,
            audit_report=report,
            passed=score >= self.min_score,
        )

    def _select_auditors(self, output_type: str) -> list[str]:
        """Select relevant auditors based on output type."""
        if output_type == "code":
            return self.AUDIT_PIPELINE  # All auditors
        elif output_type == "text":
            return [
                "logic_flaw",
                "hallucination_hunter",
                "consistency_guard",
                "output_verifier",
            ]
        else:  # analysis, general
            return [
                "logic_flaw",
                "hallucination_hunter",
                "consistency_guard",
                "edge_case_hunter",
                "output_verifier",
            ]

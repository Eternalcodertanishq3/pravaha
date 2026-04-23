"""Base Agent — Abstract foundation for all swarm agents.

Every agent in the Pravaha swarm inherits from this class. It provides:
- Unified async execute() interface with structured AgentOutput
- Token budget tracking and enforcement
- Shared context access via SharedContext
- Sampling parameter customization per agent role
- Task-type routing via can_handle()

Phase 5: Swarm agent architecture — the core building block
for the 32-agent self-healing inference pipeline.
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentOutput:
    """Structured output from any agent execution.

    Attributes:
        role: The agent's role identifier.
        output: The generated text content.
        tokens_used: Number of tokens consumed during generation.
        duration_ms: Wall-clock time for this agent's execution.
        confidence: Self-reported confidence score (0.0 to 1.0).
        metadata: Arbitrary key-value metadata (scores, categories, etc.).
        issues: List of issues found (for audit agents).
        patches: List of patches applied (for PatchApplier).
    """

    role: str
    output: str
    tokens_used: int = 0
    duration_ms: float = 0.0
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    issues: list[dict[str, Any]] = field(default_factory=list)
    patches: list[str] = field(default_factory=list)


@dataclass
class SharedContext:
    """Cross-agent shared context store.

    Mutable state accessible by all agents in a pipeline.
    Thread-safe access is managed by the SwarmOrchestrator.
    """

    task: str = ""
    plan: str = ""
    research: str = ""
    code: str = ""
    output: str = ""
    reasoning: str = ""
    tests: str = ""
    context_summary: str = ""
    merged_output: str = ""
    patched_output: str = ""
    feedback: str = ""
    task_type: str = ""
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    agent_outputs: dict[str, AgentOutput] = field(default_factory=dict)
    audit_reports: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all 32 swarm agents.

    Subclasses MUST implement:
        - role: str — unique agent identifier
        - priority: int — 0=worker, 1=senior, 2=orchestrator
        - max_tokens: int — token budget per call
        - temperature: float — sampling temperature
        - system_prompt: str — the agent's identity and instructions
        - async run(task, context, engine) -> AgentOutput
        - can_handle(task_type) -> bool

    The base class provides:
        - build_prompt(): constructs the full prompt from system + context + task
        - get_sampling_params(): returns SamplingParams for this agent's config
        - _generate(): helper to call the engine and track token usage
        - _generate_json(): helper that parses JSON from model output
    """

    role: str = "base"
    priority: int = 0
    max_tokens: int = 1024
    temperature: float = 0.5
    system_prompt: str = ""
    model_override: Optional[str] = None

    # Tracks cumulative usage across all calls
    _total_tokens: int = 0
    _total_calls: int = 0
    _total_duration_ms: float = 0.0

    def __init__(self) -> None:
        self._total_tokens = 0
        self._total_calls = 0
        self._total_duration_ms = 0.0

    @abstractmethod
    async def run(
        self,
        task: str,
        context: SharedContext,
        engine: Any,
    ) -> AgentOutput:
        """Execute this agent's primary function.

        Args:
            task: The user's original task or sub-task assigned to this agent.
            context: Shared mutable context accessible by all pipeline agents.
            engine: The AsyncPravahaEngine instance for text generation.

        Returns:
            AgentOutput with the agent's structured result.
        """
        ...

    @abstractmethod
    def can_handle(self, task_type: str) -> bool:
        """Whether this agent is relevant for the given task type.

        Args:
            task_type: One of 'code', 'research', 'writing', 'analysis',
                       'math', 'translation', 'general', 'multi_agent_output'.

        Returns:
            True if this agent should participate in the pipeline.
        """
        ...

    def get_sampling_params(self) -> Any:
        """Build SamplingParams tuned for this agent's role."""
        from pravaha.decoder.sampling import SamplingParams

        return SamplingParams(
            temperature=self.temperature,
            top_k=50 if self.temperature > 0.3 else 10,
            top_p=0.9 if self.temperature > 0.3 else 0.95,
            max_new_tokens=self.max_tokens,
            repetition_penalty=1.1,
        )

    def build_prompt(self, task: str, context: SharedContext) -> str:
        """Construct the full prompt: system + relevant context + task.

        The prompt is structured as:
            [SYSTEM] <system_prompt>
            [CONTEXT] <relevant context from shared state>
            [TASK] <the specific task for this agent>

        Subclasses can override to customize context injection.
        """
        parts = [f"[SYSTEM]\n{self.system_prompt}"]

        # Inject relevant context based on what's available
        ctx_parts = []
        if context.plan:
            ctx_parts.append(f"Plan:\n{context.plan[:500]}")
        if context.research:
            ctx_parts.append(f"Research:\n{context.research[:500]}")
        if context.code:
            ctx_parts.append(f"Code:\n{context.code[:1000]}")
        if context.output:
            ctx_parts.append(f"Current Output:\n{context.output[:500]}")
        if context.context_summary:
            ctx_parts.append(f"Context Summary:\n{context.context_summary[:300]}")

        if ctx_parts:
            parts.append("[CONTEXT]\n" + "\n\n".join(ctx_parts))

        parts.append(f"[TASK]\n{task}")

        return "\n\n".join(parts)

    async def _generate(
        self,
        prompt: str,
        engine: Any,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text using the inference engine.

        Tracks token usage and duration for observability.

        Args:
            prompt: The full prompt to send to the model.
            engine: The AsyncPravahaEngine instance.
            max_tokens: Override for this agent's default max_tokens.

        Returns:
            The generated text string.
        """
        from pravaha.decoder.sampling import SamplingParams

        params = SamplingParams(
            temperature=self.temperature,
            top_k=50 if self.temperature > 0.3 else 10,
            top_p=0.9 if self.temperature > 0.3 else 0.95,
            max_new_tokens=max_tokens or self.max_tokens,
            repetition_penalty=1.1,
        )

        tokens: list[str] = []
        async for token in engine.generate(prompt, params):
            tokens.append(token)

        output = "".join(tokens)
        self._total_tokens += len(tokens)
        self._total_calls += 1

        return output

    async def _generate_json(
        self,
        prompt: str,
        engine: Any,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Generate and parse JSON output from the model.

        Falls back to wrapping raw text in a dict if parsing fails.
        """
        import json

        raw = await self._generate(prompt, engine, max_tokens)

        # Try to extract JSON from the output
        try:
            # Handle case where model wraps JSON in markdown code fence
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            return json.loads(raw)
        except (json.JSONDecodeError, IndexError):
            # Fallback: return raw text in a structured wrapper
            return {"raw_output": raw, "parse_error": True}

    def get_stats(self) -> dict[str, Any]:
        """Return cumulative usage statistics for this agent."""
        return {
            "name": self.role,
            "priority": self.priority,
            "total_tokens": self._total_tokens,
            "total_calls": self._total_calls,
            "total_duration_ms": self._total_duration_ms,
            "avg_tokens_per_call": (
                self._total_tokens / self._total_calls
                if self._total_calls > 0
                else 0
            ),
        }

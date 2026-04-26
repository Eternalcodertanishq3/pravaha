"""Base Agent — True autonomous agent with ReAct loop, real tool access,
and persistent memory. NOT a prompt wrapper.

The ReAct (Reason + Act) loop:

THOUGHT: What do I need to do? What information do I have?
ACTION:  Which tool should I call? With what args?
OBSERVE: What did the tool return?
THOUGHT: Does this answer my question or do I need more?
...repeat up to max_steps...
ANSWER:  Final response to the task.

This loop is what makes an agent truly autonomous — it separates
Pravaha from "a swarm of system prompts."
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pravaha.swarm.memory.memory_store import MemoryStore

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────


@dataclass
class ToolCall:
    """A parsed tool invocation from agent output."""

    tool_name: str
    args: dict[str, Any]


@dataclass
class ReActStep:
    """A single step in the ReAct loop."""

    thought: str = ""
    action: ToolCall | None = None
    observation: str = ""
    is_final_answer: bool = False
    answer: str = ""


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
        trajectory: ReAct steps taken during execution.
    """

    role: str
    output: str
    tokens_used: int = 0
    duration_ms: float = 0.0
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    issues: list[dict[str, Any]] = field(default_factory=list)
    patches: list[str] = field(default_factory=list)
    trajectory: list[ReActStep] = field(default_factory=list)


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


# ── Agent Memory Wrapper ─────────────────────────────────────────────


class AgentMemory:
    """Per-agent memory wrapper around the shared MemoryStore."""

    def __init__(self, store: MemoryStore, agent_role: str) -> None:
        self._store = store
        self._role = agent_role

    def store(self, fact: str, importance: float = 0.5) -> None:
        """Store a fact in this agent's memory."""
        key = f"fact_{hash(fact) % 100000}"
        self._store.put(self._role, key, fact, importance)

    def get(self, key: str) -> str | None:
        """Retrieve a specific memory by key."""
        return self._store.get(self._role, key)

    def get_recent(self, limit: int = 5) -> list[str]:
        """Get most recently accessed memories."""
        return self._store.get_recent(self._role, limit)

    def get_important(self, min_importance: float = 0.7) -> list[str]:
        """Get high-importance memories."""
        return self._store.get_important(self._role, min_importance)


# ── Base Agent ────────────────────────────────────────────────────────


class BaseAgent(ABC):
    """Abstract base class for all swarm agents.

    True autonomous agent with ReAct loop, real tool access,
    and persistent memory.

    Subclasses MUST implement:
        - role: str — unique agent identifier
        - can_handle(task_type) -> bool

    Subclasses MAY override:
        - run() for fully custom execution
        - available_tools: list of tool names this agent can use
        - max_react_steps: max ReAct iterations before forcing answer
    """

    role: str = "base"
    priority: int = 0
    max_tokens: int = 1024
    temperature: float = 0.5
    system_prompt: str = ""
    model_override: str | None = None
    max_react_steps: int = 5

    # Tools this agent can access (set by subclasses)
    available_tools: list[str] = []

    def __init__(self) -> None:
        self._total_tokens: int = 0
        self._total_calls: int = 0
        self._total_duration_ms: float = 0.0
        self._tool_registry: Any | None = None
        self._memory: AgentMemory | None = None

    def attach_tools(self, registry: Any) -> None:
        """Attach a tool registry for real tool execution."""
        self._tool_registry = registry

    def attach_memory(self, memory: AgentMemory) -> None:
        """Attach persistent memory for this agent."""
        self._memory = memory

    # ── ReAct Loop ────────────────────────────────────────────────

    async def run_react(
        self,
        task: str,
        context: SharedContext,
        engine: Any,
    ) -> AgentOutput:
        """Execute the ReAct loop: Thought → Action → Observation → ...

        The agent generates structured output with Thought/Action/Final Answer
        markers. The base class parses and executes tool calls automatically.
        """
        t0 = time.time()
        react_prompt = self._build_react_prompt(task, context)
        trajectory: list[ReActStep] = []

        for step_num in range(self.max_react_steps):
            output = await self._generate(react_prompt, engine)
            parsed = self._parse_react_output(output)

            if parsed.is_final_answer:
                duration = (time.time() - t0) * 1000
                self._total_duration_ms += duration
                return self._build_output(
                    task=task,
                    answer=parsed.answer,
                    trajectory=trajectory,
                    context=context,
                    duration_ms=duration,
                )

            trajectory.append(parsed)

            # Execute the tool call if present
            if parsed.action and self._tool_registry:
                try:
                    observation = await self._tool_registry.execute(
                        tool_name=parsed.action.tool_name,
                        args=parsed.action.args,
                    )
                except Exception as e:
                    observation = f"ERROR: Tool execution failed: {e}"

                parsed.observation = observation

                # Append observation to prompt using XML delimiters
                react_prompt += (
                    f"\n<thought>{parsed.thought}</thought>\n"
                    f"<action>{parsed.action.tool_name}</action>\n"
                    f"<args>{json.dumps(parsed.action.args, ensure_ascii=False)}</args>\n"
                    f"<observation>{observation[:500]}</observation>\n\n"
                    f"<thought>"
                )
            else:
                # No tool call — force final answer on next step
                react_prompt += (
                    f"\n{output}\n"
                    f"You must now provide your final answer.\n"
                    f"<answer>"
                )

        # Max steps reached — synthesize from trajectory
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration
        final = await self._synthesize_answer(task, trajectory, context, engine)
        return self._build_output(task, final, trajectory, context, duration)

    def _build_react_prompt(self, task: str, context: SharedContext) -> str:
        """Build ReAct prompt using XML-style delimiters for reliable parsing."""
        tools_desc = ""
        if self._tool_registry:
            tools = self._tool_registry.get_available(self.available_tools)
            ranked = self._rank_tools(task, tools)
            tools_desc = "\n".join(
                f"- {t.name}: {t.description}"
                f"\n  args: {t.arg_schema}"
                for t in ranked
            )

        memory_ctx = ""
        if self._memory:
            recent = self._memory.get_recent(limit=3)
            if recent:
                memory_ctx = "\nPAST CONTEXT:\n" + "\n".join(
                    f"\u2022 {m}" for m in recent
                )

        return (
            f"<system>\n{self.system_prompt}\n</system>\n\n"
            f"<tools>\n{tools_desc or 'None available'}\n</tools>\n\n"
            f"<context>\n{self._get_context_summary(context)}"
            f"{memory_ctx}\n</context>\n\n"
            f"<task>\n{task}\n</task>\n\n"
            "Respond in EXACTLY this format. Never deviate:\n"
            "<thought>your reasoning here</thought>\n"
            "<action>tool_name</action>\n"
            '<args>{"key": "value"}</args>\n\n'
            "OR if you have enough information:\n"
            "<thought>your final reasoning</thought>\n"
            "<answer>your complete final answer</answer>\n\n"
            "<thought>"
        )

    @staticmethod
    def _rank_tools(task: str, tools: list) -> list:  # type: ignore[type-arg]
        """Sort tools by relevance to the task for better agent decisions."""
        task_lower = task.lower()
        relevance: dict[str, int] = {
            "execute_python": 10 if any(
                kw in task_lower for kw in
                ["run", "test", "execute", "verify", "check", "compute"]
            ) else 3,
            "web_search": 10 if any(
                kw in task_lower for kw in
                ["find", "search", "look up", "what is", "latest", "current"]
            ) else 2,
            "read_file": 10 if any(
                kw in task_lower for kw in
                ["read", "analyze", "review", "check", "file", "code"]
            ) else 2,
            "fetch_url": 10 if any(
                kw in task_lower for kw in
                ["http", "url", "website", "page", "download"]
            ) else 1,
            "memory": 10 if any(
                kw in task_lower for kw in
                ["remember", "previous", "last time", "before", "past"]
            ) else 3,
        }
        return sorted(
            tools,
            key=lambda t: relevance.get(t.name, 1),
            reverse=True,
        )

    def _get_context_summary(self, context: SharedContext) -> str:
        """Extract relevant context fields into a summary string."""
        parts: list[str] = []
        if context.plan:
            parts.append(f"Plan:\n{context.plan[:500]}")
        if context.research:
            parts.append(f"Research:\n{context.research[:500]}")
        if context.code:
            parts.append(f"Code:\n{context.code[:1000]}")
        if context.output:
            parts.append(f"Current Output:\n{context.output[:500]}")
        if context.context_summary:
            parts.append(f"Context Summary:\n{context.context_summary[:300]}")
        if context.feedback:
            parts.append(f"Feedback:\n{context.feedback[:300]}")
        return "\n\n".join(parts) if parts else "No prior context."

    def _parse_react_output(self, text: str) -> ReActStep:
        """Parse XML-delimited ReAct output. Robust against content in tags."""
        step = ReActStep()

        # Extract thought (first one only)
        thought_match = re.search(
            r"<thought>(.*?)</thought>",
            text, re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            step.thought = thought_match.group(1).strip()

        # Check for final answer first (highest priority)
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            text, re.DOTALL | re.IGNORECASE,
        )
        if answer_match:
            step.is_final_answer = True
            step.answer = answer_match.group(1).strip()
            return step

        # Legacy fallback: "Final Answer:" for backward compatibility
        if "Final Answer:" in text and "<action>" not in text:
            step.is_final_answer = True
            step.answer = text.split("Final Answer:")[-1].strip()
            return step

        # Extract action + args
        action_match = re.search(
            r"<action>(.*?)</action>",
            text, re.DOTALL | re.IGNORECASE,
        )
        args_match = re.search(
            r"<args>(.*?)</args>",
            text, re.DOTALL | re.IGNORECASE,
        )

        if action_match:
            tool_name = action_match.group(1).strip()
            args: dict[str, Any] = {}
            if args_match:
                try:
                    raw_args = args_match.group(1).strip()
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {"raw": args_match.group(1).strip()}
            step.action = ToolCall(tool_name=tool_name, args=args)
        else:
            # Legacy fallback: tool_name({"arg": "val"}) format
            step.action = self._parse_tool_call_legacy(text)

        # Fallback: if no structured tags found but text has content,
        # treat entire output as final answer to avoid infinite loops
        if not step.action and not step.thought and text.strip():
            step.is_final_answer = True
            step.answer = text.strip()

        return step

    @staticmethod
    def _parse_tool_call_legacy(text: str) -> ToolCall | None:
        """Legacy parser for 'tool_name({"arg": "val"})' format."""
        # Try Action: prefix format
        action_text = text
        if "Action:" in text:
            action_text = text.split("Action:")[-1].split("Observation:")[0].strip()
        match = re.match(r"(\w+)\s*\((.+)\)", action_text.strip(), re.DOTALL)
        if match:
            name = match.group(1)
            try:
                args = json.loads(match.group(2))
                return ToolCall(tool_name=name, args=args)
            except json.JSONDecodeError:
                return ToolCall(tool_name=name, args={"raw": match.group(2)})
        return None

    def _build_output(
        self,
        task: str,
        answer: str,
        trajectory: list[ReActStep],
        context: SharedContext,
        duration_ms: float = 0.0,
    ) -> AgentOutput:
        """Build the final AgentOutput from a completed ReAct loop."""
        # Store in memory if available
        if self._memory:
            self._memory.store(
                f"Task: {task[:80]} → Answer: {answer[:80]}",
                importance=0.6,
            )

        return AgentOutput(
            role=self.role,
            output=answer,
            tokens_used=self._total_tokens,
            duration_ms=duration_ms,
            confidence=min(1.0, 0.5 + 0.1 * len(trajectory)),
            trajectory=trajectory,
            metadata={"react_steps": len(trajectory)},
        )

    async def _synthesize_answer(
        self,
        task: str,
        trajectory: list[ReActStep],
        context: SharedContext,
        engine: Any,
    ) -> str:
        """Synthesize a final answer from a trajectory when max steps reached."""
        observations = [
            s.observation for s in trajectory if s.observation
        ]
        if observations:
            synthesis_prompt = (
                f"Based on the following observations, provide a final answer "
                f"for the task: {task}\n\n"
                + "\n".join(f"- {o[:200]}" for o in observations)
            )
            return await self._generate(synthesis_prompt, engine)
        return "Unable to complete task within step limit."

    # ── Default run() ─────────────────────────────────────────────

    async def run(
        self,
        task: str,
        context: SharedContext,
        engine: Any,
    ) -> AgentOutput:
        """Default run() uses ReAct if tools are available,
        falls back to direct generation if not.

        Subclasses can override for specialized behavior.
        """
        if self._tool_registry and self.available_tools:
            return await self.run_react(task, context, engine)

        # Direct generation (legacy mode for simple agents)
        t0 = time.time()
        prompt = self.build_prompt(task, context)
        output = await self._generate(prompt, engine)
        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration

        if self._memory:
            self._memory.store(f"Task: {task[:100]} → {output[:100]}")

        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
        )

    @abstractmethod
    def can_handle(self, task_type: str) -> bool:
        """Whether this agent is relevant for the given task type."""
        ...

    # ── Prompt Building ───────────────────────────────────────────

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
        """Construct the full prompt: system + relevant context + task."""
        parts = [f"[SYSTEM]\n{self.system_prompt}"]

        ctx_summary = self._get_context_summary(context)
        if ctx_summary != "No prior context.":
            parts.append(f"[CONTEXT]\n{ctx_summary}")

        parts.append(f"[TASK]\n{task}")
        return "\n\n".join(parts)

    # ── Generation Helpers ────────────────────────────────────────

    async def _generate(
        self,
        prompt: str,
        engine: Any,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text using the inference engine."""
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
        """Generate and parse JSON output from the model."""
        raw = await self._generate(prompt, engine, max_tokens)

        try:
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            return json.loads(raw)
        except (json.JSONDecodeError, IndexError):
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
            "has_tools": bool(self.available_tools),
            "has_memory": self._memory is not None,
        }

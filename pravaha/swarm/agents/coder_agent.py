"""Coder Agent — Production-ready code generation.

Writes clean, well-documented, type-hinted code. Follows best
practices, includes docstrings, and structures output for
direct use. Consumes plans from PlannerAgent and research from
ResearcherAgent.

Priority: 1 (senior worker).
"""

from __future__ import annotations

import time
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class CoderAgent(BaseAgent):
    """Generates clean, production-ready code with documentation."""

    role = "coder"
    priority = 1
    max_tokens = 2048
    temperature = 0.2
    system_prompt = (
        "You are a senior software engineer. Write clean, well-documented, "
        "production-ready code.\n\n"
        "Requirements:\n"
        "1. Include complete type hints on ALL function signatures\n"
        "2. Add Google-style docstrings to every class and public method\n"
        "3. Follow language-specific best practices and idioms\n"
        "4. Handle errors explicitly — never use bare except\n"
        "5. Use descriptive variable and function names\n"
        "6. Add inline comments explaining WHY, not WHAT\n"
        "7. Include necessary imports at the top\n"
        "8. Structure code for testability (dependency injection)\n"
        "9. Follow SOLID principles where applicable\n"
        "10. If writing Python, target 3.11+ with modern syntax\n\n"
        "Output ONLY the code. No explanations before or after.\n"
        "Wrap the code in appropriate markdown code fences with language tag."
    )

    async def run(self, task: str, context: SharedContext, engine: Any) -> AgentOutput:
        t0 = time.time()
        prompt = self.build_prompt(task, context)
        output = await self._generate(prompt, engine)

        # Extract code from markdown fences if present
        code = self._extract_code(output)
        context.code = code

        # Estimate code quality metrics
        lines = code.split("\n")
        loc = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
        has_types = "def " in code and "->" in code
        has_docs = '"""' in code or "'''" in code

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration

        return AgentOutput(
            role=self.role,
            output=code,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=0.85 if has_types and has_docs else 0.65,
            metadata={
                "lines_of_code": loc,
                "has_type_hints": has_types,
                "has_docstrings": has_docs,
                "language": self._detect_language(code),
            },
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"code", "function", "class", "script", "module"}

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract code from markdown fences, or return as-is."""
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                code_block = parts[1]
                # Remove language tag from first line
                lines = code_block.split("\n")
                if lines and not lines[0].strip().startswith(("def ", "class ", "import ", "from ")):
                    lines = lines[1:]
                return "\n".join(lines).strip()
        return text.strip()

    @staticmethod
    def _detect_language(code: str) -> str:
        """Simple heuristic to detect programming language."""
        if "def " in code and "import " in code:
            return "python"
        if "function " in code or "const " in code or "=>" in code:
            return "javascript"
        if "fn " in code and "let " in code:
            return "rust"
        if "func " in code and "package " in code:
            return "go"
        return "unknown"

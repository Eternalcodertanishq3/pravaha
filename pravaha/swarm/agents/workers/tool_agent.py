"""Tool Agent — External tool orchestration."""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class ToolAgent(BaseAgent):
    role = "tool"
    priority = 2
    max_tokens = 1024
    temperature = 0.1
    available_tools = [
        "execute_python", "read_file", "web_search",
        "fetch_url", "run_shell",
    ]

    system_prompt = (
        "You are a tool orchestration agent. Use tools to gather\n"
        "information or perform actions as needed. Report results."
    )

    def can_handle(self, task_type: str) -> bool:
        return True

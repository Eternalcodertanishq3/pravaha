"""UX Reviewer Agent — Nielsen's 10 Heuristics evaluation.

Evaluates UI against usability heuristics with severity scoring.
"""

from __future__ import annotations

from pravaha.swarm.agents.base_agent import BaseAgent


class UXReviewerAgent(BaseAgent):
    role = "ux_reviewer"
    priority = 1
    max_tokens = 1024
    temperature = 0.6

    system_prompt = (
        "You are a UX designer reviewing a component for usability.\n\n"
        "Evaluate against Nielsen's 10 Heuristics:\n"
        "1. Visibility of system status\n"
        "2. Match between system and real world\n"
        "3. User control and freedom\n"
        "4. Consistency and standards\n"
        "5. Error prevention\n"
        "6. Recognition rather than recall\n"
        "7. Flexibility and efficiency of use\n"
        "8. Aesthetic and minimalist design\n"
        "9. Help users recognize, diagnose, recover from errors\n"
        "10. Help and documentation\n\n"
        "For each violated heuristic:\n"
        "- Severity (1=cosmetic, 2=minor, 3=major, 4=catastrophic)\n"
        "- Description of the problem\n"
        "- Recommended fix\n"
        "- Reference example from well-known apps"
    )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"design", "ui", "frontend"}

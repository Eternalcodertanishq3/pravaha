"""Audit Panel — Self-healing audit loop live view."""

from __future__ import annotations

from textual.widgets import Static


class AuditPanel(Static):
    """Shows audit loop status: active auditors, issues, patches."""

    DEFAULT_CSS = """
    AuditPanel { height: 4; border: solid #1a2a1a 1; padding: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.active_auditor = ""
        self.issues_found = 0
        self.patches_applied = 0
        self.iteration = 0
        self.status = "idle"

    def render(self) -> str:
        if self.status == "idle":
            return "Audit: [idle] — waiting for output to audit"
        return (
            f"Audit: [{self.active_auditor}: scanning...] "
            f"iter={self.iteration}/3  issues={self.issues_found}  "
            f"patches={self.patches_applied}  [{self.status}]"
        )

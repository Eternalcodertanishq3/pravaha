"""File Reader — Read local files with security restrictions."""

from __future__ import annotations

from pathlib import Path


class FileReader:
    """Read contents of a local file with extension whitelist."""

    name = "read_file"
    description = "Read contents of a local file"
    arg_schema = '{"path": "string", "max_bytes": 8192}'

    ALLOWED_EXTENSIONS = {
        ".py", ".js", ".ts", ".tsx", ".jsx",
        ".md", ".txt", ".json", ".yaml", ".yml",
        ".toml", ".cfg", ".ini", ".csv", ".html",
        ".css", ".rs", ".go", ".java", ".c", ".h",
        ".sh", ".bat", ".dockerfile",
    }

    MAX_OUTPUT_BYTES = 8192

    def execute(self, path: str, max_bytes: int = 8192) -> dict:
        """Read file contents."""
        p = Path(path)
        if not p.exists():
            return {"error": f"File not found: {path}", "success": False}
        if p.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            return {
                "error": f"File type not allowed: {p.suffix}",
                "success": False,
            }
        max_bytes = min(max_bytes, self.MAX_OUTPUT_BYTES)
        try:
            content = p.read_text(encoding="utf-8", errors="replace")[:max_bytes]
            return {
                "content": content,
                "lines": content.count("\n") + 1,
                "size_bytes": p.stat().st_size,
                "success": True,
            }
        except Exception as e:
            return {"error": str(e), "success": False}

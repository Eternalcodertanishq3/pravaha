"""Rollback Manager — One-command automated release rollback script.

Validates service health post-rollback and reverts to previous git release tag or commit.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class RollbackManager:
    """Automated rollback executor for Pravaha deployments."""

    def __init__(self, repo_path: str = ".", target_url: str = "http://localhost:8000") -> None:
        self.repo_path = Path(repo_path).resolve()
        self.target_url = target_url

    def get_current_commit(self) -> str:
        """Get short commit hash of current HEAD."""
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def get_previous_tag() -> str | None:
        """Find previous release tag in git history."""
        try:
            result = subprocess.run(
                ["git", "tag", "--sort=-creatordate"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            tags = [t.strip() for t in result.stdout.splitlines() if t.strip()]
            return tags[1] if len(tags) > 1 else (tags[0] if tags else None)
        except Exception:
            return None

    def execute_rollback(self, target_ref: str | None = None) -> bool:
        """Rollback git repository to target tag/commit and verify health."""
        current = self.get_current_commit()
        ref = target_ref or self.get_previous_tag() or "HEAD~1"

        logger.info(f"Initiating rollback from {current} to {ref}...")

        try:
            # Checkout target ref
            subprocess.run(["git", "checkout", ref], cwd=self.repo_path, check=True)
            logger.info(f"Git checkout successful: checked out {ref}.")

            # Verify health probe
            health_ok = self.check_health(retries=5, delay_s=2)
            if health_ok:
                logger.info(f"Rollback to {ref} COMPLETED SUCCESSFULLY.")
                return True
            else:
                logger.error(f"Health check failed after rollback to {ref}.")
                return False

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def check_health(self, retries: int = 5, delay_s: int = 2) -> bool:
        """Check target service readiness endpoint."""
        url = f"{self.target_url}/health"
        for i in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "RollbackManager"})
                with urllib.request.urlopen(req, timeout=3) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                time.sleep(delay_s)
        return False


def main():
    parser = argparse.ArgumentParser(description="Pravaha One-Command Rollback Tool")
    parser.add_argument("--ref", default=None, help="Target git tag/commit to rollback to")
    parser.add_argument("--url", default="http://localhost:8000", help="Service base URL")

    args = parser.parse_args()
    manager = RollbackManager(target_url=args.url)
    success = manager.execute_rollback(target_ref=args.ref)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

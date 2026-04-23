"""Tests for the pipeline definitions."""

from __future__ import annotations

import pytest

from pravaha.swarm.pipeline import BUILTIN_PIPELINES


class TestPipelines:
    def test_builtin_count(self) -> None:
        assert len(BUILTIN_PIPELINES) >= 5

    def test_plan_execute_audit_exists(self) -> None:
        assert "plan-execute-audit" in BUILTIN_PIPELINES

    def test_all_pipelines_have_workers(self) -> None:
        for name, pipe in BUILTIN_PIPELINES.items():
            assert len(pipe.worker_steps) > 0, f"{name} has no workers"

    def test_all_pipelines_have_auditors(self) -> None:
        for name, pipe in BUILTIN_PIPELINES.items():
            assert len(pipe.audit_steps) > 0, f"{name} has no auditors"

    def test_all_workers_are_valid(self) -> None:
        from pravaha.swarm.agents import ALL_AGENTS
        for name, pipe in BUILTIN_PIPELINES.items():
            for worker in pipe.worker_steps:
                assert worker in ALL_AGENTS, f"Pipeline '{name}' references unknown agent '{worker}'"

    def test_pipeline_descriptions(self) -> None:
        for name, pipe in BUILTIN_PIPELINES.items():
            assert pipe.description, f"{name} has no description"

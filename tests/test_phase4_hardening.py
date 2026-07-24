"""Tests for Phase 4 Hardening features: RBAC, DockerSandbox, Rollback, GDPR, Factory."""

import pytest
from pravaha.serving.rbac import RBACManager, Role
from pravaha.swarm.tools.docker_sandbox import DockerSandbox
from scripts.rollback import RollbackManager
from pravaha.engine.factory import EngineFactory
from pravaha.config.engine_config import EngineConfig


def test_rbac_hierarchy():
    """Verify RBAC role hierarchy permissions."""
    assert RBACManager.has_permission(Role.ADMIN, Role.USER) is True
    assert RBACManager.has_permission(Role.ADMIN, Role.OPERATOR) is True
    assert RBACManager.has_permission(Role.ADMIN, Role.ADMIN) is True

    assert RBACManager.has_permission(Role.OPERATOR, Role.USER) is True
    assert RBACManager.has_permission(Role.OPERATOR, Role.ADMIN) is False

    assert RBACManager.has_permission(Role.USER, Role.ADMIN) is False
    assert RBACManager.has_permission(Role.USER, Role.OPERATOR) is False


def test_docker_sandbox_fallback():
    """Verify DockerSandbox command execution and fallback."""
    sandbox = DockerSandbox(allow_network=False)
    result = sandbox.execute_command(["python", "-c", "print('hello_sandbox')"], timeout_s=5)

    assert result["success"] is True
    assert "hello_sandbox" in result["stdout"]
    assert "sandbox_type" in result


def test_rollback_manager_health_check():
    """Verify RollbackManager health check method handles unreachable endpoints safely."""
    manager = RollbackManager(target_url="http://127.0.0.1:59999")  # Unreachable port
    assert manager.check_health(retries=1, delay_s=0.1) is False


def test_engine_factory_construction():
    """Verify EngineFactory creates all subsystem instances."""
    config = EngineConfig.default()
    subs = EngineFactory.build_subsystems(config, device="cpu")

    assert "tokenizer" in subs
    assert "model" in subs
    assert "scheduler" in subs
    assert "session_cache" in subs
    assert subs["num_blocks"] > 0

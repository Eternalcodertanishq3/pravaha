"""Tests for security agents — static regex scanning."""

from __future__ import annotations

import pytest

from pravaha.swarm.agents.base_agent import SharedContext


class TestSecurityAudit:
    """Test SecurityAuditAgent with CVSS scoring."""

    @pytest.mark.asyncio
    async def test_detects_eval(self) -> None:
        from pravaha.swarm.agents.security.security_audit_agent import SecurityAuditAgent
        agent = SecurityAuditAgent()
        ctx = SharedContext()
        ctx.code = "result = eval(user_input)\n"
        result = await agent.run("audit", ctx, None)
        assert len(result.issues) > 0
        assert any(i["cvss"] >= 9.0 for i in result.issues)

    @pytest.mark.asyncio
    async def test_detects_hardcoded_secret(self) -> None:
        from pravaha.swarm.agents.security.security_audit_agent import SecurityAuditAgent
        agent = SecurityAuditAgent()
        ctx = SharedContext()
        ctx.code = 'password = "hunter2"\n'
        result = await agent.run("audit", ctx, None)
        assert any(i["id"] == "hardcoded_password" for i in result.issues)


class TestInjectionScanner:
    @pytest.mark.asyncio
    async def test_detects_sql_injection(self) -> None:
        from pravaha.swarm.agents.security.injection_scanner_agent import InjectionScannerAgent
        agent = InjectionScannerAgent()
        ctx = SharedContext()
        ctx.code = 'db.execute("SELECT * FROM users WHERE id=" + user_id)\n'
        result = await agent.run("audit", ctx, None)
        assert any(i["id"] == "sql_injection" for i in result.issues)


class TestSecretsScanner:
    @pytest.mark.asyncio
    async def test_detects_aws_key(self) -> None:
        from pravaha.swarm.agents.security.secrets_scanner_agent import SecretsScannerAgent
        agent = SecretsScannerAgent()
        ctx = SharedContext()
        ctx.code = 'aws_key = "AKIAIOSFODNN7EXAMPLE"\n'
        result = await agent.run("audit", ctx, None)
        assert any(i["id"] == "aws_key" for i in result.issues)

    @pytest.mark.asyncio
    async def test_clean_code(self) -> None:
        from pravaha.swarm.agents.security.secrets_scanner_agent import SecretsScannerAgent
        agent = SecretsScannerAgent()
        ctx = SharedContext()
        ctx.code = 'import os\nkey = os.environ["API_KEY"]\n'
        result = await agent.run("audit", ctx, None)
        assert len(result.issues) == 0


class TestCryptoAudit:
    @pytest.mark.asyncio
    async def test_detects_md5(self) -> None:
        from pravaha.swarm.agents.security.crypto_audit_agent import CryptoAuditAgent
        agent = CryptoAuditAgent()
        ctx = SharedContext()
        ctx.code = 'import hashlib\nhash = hashlib.md5(data).hexdigest()\n'
        result = await agent.run("audit", ctx, None)
        assert any(i["id"] == "weak_hash_md5" for i in result.issues)

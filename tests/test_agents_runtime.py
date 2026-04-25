"""Runtime tests for agents with MockEngine — verifies actual agent logic."""

from __future__ import annotations

import pytest

from pravaha.swarm.agents.base_agent import AgentOutput, SharedContext


class MockTokenizer:
    """Minimal tokenizer for testing."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)


class MockEngine:
    """Mock engine that returns a canned response for agent testing.

    Simulates the generate() async generator interface that agents expect.
    """

    def __init__(self, returns: str = "{}") -> None:
        self.tokenizer = MockTokenizer()
        self._returns = returns

    async def generate(self, prompt: str, params: object = None):
        for token in self._returns.split():
            yield token + " "


class TestSyntaxAuditFindsRealBug:
    """SyntaxAuditAgent must find real issues via static scan."""

    @pytest.mark.asyncio
    async def test_finds_eval_usage(self) -> None:
        from pravaha.swarm.agents.auditors.syntax_audit_agent import SyntaxAuditAgent

        agent = SyntaxAuditAgent()
        ctx = SharedContext()
        ctx.code = "result = eval(user_input)\n"
        result = await agent.run("check code", ctx, None)
        assert len(result.issues) > 0
        assert any(i.get("id") == "eval_usage" for i in result.issues)

    @pytest.mark.asyncio
    async def test_finds_bare_except(self) -> None:
        from pravaha.swarm.agents.auditors.syntax_audit_agent import SyntaxAuditAgent

        agent = SyntaxAuditAgent()
        ctx = SharedContext()
        ctx.code = "try:\n    x = 1\nexcept:\n    pass\n"
        result = await agent.run("check code", ctx, None)
        assert any(i.get("id") == "bare_except" for i in result.issues)


class TestSecurityAuditFindsHardcodedSecret:
    @pytest.mark.asyncio
    async def test_finds_hardcoded_password(self) -> None:
        from pravaha.swarm.agents.security.security_audit_agent import SecurityAuditAgent

        agent = SecurityAuditAgent()
        ctx = SharedContext()
        ctx.code = 'password = "supersecret123"\n'
        result = await agent.run("check", ctx, None)
        assert len(result.issues) > 0
        assert any(i.get("cvss", 0) >= 5.0 for i in result.issues)

    @pytest.mark.asyncio
    async def test_finds_eval(self) -> None:
        from pravaha.swarm.agents.security.security_audit_agent import SecurityAuditAgent

        agent = SecurityAuditAgent()
        ctx = SharedContext()
        ctx.code = "data = eval(request.body)\n"
        result = await agent.run("check", ctx, None)
        assert len(result.issues) > 0


class TestPatchApplierMarksPatched:
    @pytest.mark.asyncio
    async def test_patched_output(self) -> None:
        from pravaha.swarm.agents.auditors.patch_applier_agent import PatchApplierAgent

        agent = PatchApplierAgent()
        ctx = SharedContext()
        ctx.code = "x = 1\n"
        ctx.audit_reports = [{"issues": [{"description": "unused variable x"}]}]
        engine = MockEngine(returns="x = 1  # PATCHED: fixed unused variable\n")
        result = await agent.run("fix", ctx, engine)
        # PatchApplier should return something non-empty
        assert result.output


class TestInjectionScannerFindsSQL:
    @pytest.mark.asyncio
    async def test_sql_injection(self) -> None:
        from pravaha.swarm.agents.security.injection_scanner_agent import InjectionScannerAgent

        agent = InjectionScannerAgent()
        ctx = SharedContext()
        ctx.code = 'db.execute("SELECT * FROM users WHERE id=" + user_id)\n'
        result = await agent.run("audit", ctx, None)
        assert any(i.get("id") == "sql_injection" for i in result.issues)


class TestCryptoAuditFindsMD5:
    @pytest.mark.asyncio
    async def test_md5_detection(self) -> None:
        from pravaha.swarm.agents.security.crypto_audit_agent import CryptoAuditAgent

        agent = CryptoAuditAgent()
        ctx = SharedContext()
        ctx.code = "import hashlib\nhash = hashlib.md5(data).hexdigest()\n"
        result = await agent.run("audit", ctx, None)
        assert any(i.get("id") == "weak_hash_md5" for i in result.issues)


class TestAccessibilityFindsMissingAlt:
    @pytest.mark.asyncio
    async def test_missing_alt(self) -> None:
        from pravaha.swarm.agents.design.accessibility_agent import AccessibilityAgent

        agent = AccessibilityAgent()
        ctx = SharedContext()
        ctx.code = '<img src="photo.jpg">\n'
        result = await agent.run("audit", ctx, None)
        assert any(i.get("id") == "missing_alt" for i in result.issues)


class TestSecretsEntropyDetection:
    @pytest.mark.asyncio
    async def test_high_entropy_string(self) -> None:
        from pravaha.swarm.agents.security.secrets_scanner_agent import SecretsScannerAgent

        agent = SecretsScannerAgent()
        ctx = SharedContext()
        # High-entropy string that looks like a secret
        ctx.code = 'api_key = "a3f8K9mP2qW7xR4vL6nT1bJ5cY0dH8eG"\n'
        result = await agent.run("audit", ctx, None)
        # Should flag either as a hardcoded_password or high entropy
        assert len(result.issues) > 0

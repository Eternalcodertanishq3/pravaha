"""Security Agents — 10 security-focused audit agents."""

from pravaha.swarm.agents.security.security_audit_agent import SecurityAuditAgent
from pravaha.swarm.agents.security.injection_scanner_agent import InjectionScannerAgent
from pravaha.swarm.agents.security.auth_audit_agent import AuthAuditAgent
from pravaha.swarm.agents.security.crypto_audit_agent import CryptoAuditAgent
from pravaha.swarm.agents.security.dependency_audit_agent import DependencyAuditAgent
from pravaha.swarm.agents.security.secrets_scanner_agent import SecretsScannerAgent
from pravaha.swarm.agents.security.network_security_agent import NetworkSecurityAgent
from pravaha.swarm.agents.security.privilege_audit_agent import PrivilegeAuditAgent
from pravaha.swarm.agents.security.api_security_agent import APISecurityAgent
from pravaha.swarm.agents.security.compliance_agent import ComplianceAgent

SECURITY_AGENTS: dict[str, type] = {
    "security_audit": SecurityAuditAgent,
    "injection_scanner": InjectionScannerAgent,
    "auth_audit": AuthAuditAgent,
    "crypto_audit": CryptoAuditAgent,
    "dependency_audit": DependencyAuditAgent,
    "secrets_scanner": SecretsScannerAgent,
    "network_security": NetworkSecurityAgent,
    "privilege_audit": PrivilegeAuditAgent,
    "api_security": APISecurityAgent,
    "compliance": ComplianceAgent,
}

__all__ = ["SECURITY_AGENTS"]

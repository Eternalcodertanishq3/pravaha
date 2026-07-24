# Pravāha v3.3 Plugin Development & Extension Guide

This guide describes the plugin system architecture for **Pravāha v3.3**. Pravāha's plugin engine allows staff engineers and enterprise developers to extend every layer of the system—from request authentication, custom guardrail inspection, dynamic token generation filtering, and custom ReAct agent tools, to low-level KV-cache metric exporters.

---

## 1. Plugin System Architecture Overview

Pravāha's plugin framework is built on top of Python entry points (`pravaha.plugins`), dynamic module loading, and event-driven lifecycle hooks.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Gateway Request Path                                                    │
│ Incoming Client Request $\rightarrow$ BearerAuthMiddleware $\rightarrow$ RBACManager        │
├─────────────────────────────────────────────────────────────────────────┤
│ Hook Point 1: `on_request(request: RequestContext)`                     │
│ [Plugin Inspection / Prompt Modification / Custom Header Injection]    │
├─────────────────────────────────────────────────────────────────────────┤
│ Continuous Inference & Token Generation Loop                            │
│ Continuous Scheduler $\rightarrow$ Model Inference $\rightarrow$ Token Sampler               │
├─────────────────────────────────────────────────────────────────────────┤
│ Hook Point 2: `on_token_generate(token: TokenContext)`                  │
│ [Real-time Token Interception / Safety Filtering / Stream Masking]      │
├─────────────────────────────────────────────────────────────────────────┤
│ Agent Swarm Execution Loop                                              │
│ ReAct Agent Loop (THINK $\rightarrow$ ACT $\rightarrow$ OBSERVE $\rightarrow$ ANSWER)                │
├─────────────────────────────────────────────────────────────────────────┤
│ Hook Point 3: `on_agent_step(step: AgentStepContext)`                   │
│ [Tool Interception / Custom Tool Injection / Audit Enforcement]         │
├─────────────────────────────────────────────────────────────────────────┤
│ Response Return Path                                                    │
│ Hook Point 4: `on_response(response: ResponseContext)`                  │
│ [Post-processing / Custom Metadata Injection / SHA-256 Signing]         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Design Principles

- **Zero-Copy Interception**: Context objects passed to plugin hooks use reference handles to avoid serializing prompt or token payloads.
- **Fail-Safe Isolation**: Unhandled exceptions inside plugin hooks are caught by the plugin manager, logged to `StructuredLogger`, and isolated to prevent crashing the core `AsyncPravahaEngine`.
- **RBAC & Security Scoping**: Plugins inherit security context from `RBACManager` and run custom agent tools inside `DockerSandbox`.

---

## 2. Base Plugin SDK Reference (`pravaha.plugins.base_plugin`)

All custom plugins inherit from `BasePlugin`. Below is the complete interface implementation:

```python
"""Base Plugin Abstraction Layer for Pravāha v3.3."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PluginContext:
    """Provides plugins with safe access to internal engine primitives."""

    def __init__(
        self,
        engine: Any,
        tool_registry: Any,
        memory_store: Any,
        audit_trail: Any,
        config: Dict[str, Any],
    ) -> None:
        self.engine = engine
        self.tool_registry = tool_registry
        self.memory_store = memory_store
        self.audit_trail = audit_trail
        self.config = config


class BasePlugin(ABC):
    """Abstract Base Class for all Pravāha v3.3 plugins."""

    name: str = "base-plugin"
    version: str = "1.0.0"
    description: str = "Base plugin description"
    author: str = "Pravāha Core Team"
    priority: int = 100  # Lower numbers execute earlier

    def __init__(self) -> None:
        self.context: Optional[PluginContext] = None
        self.enabled: bool = True

    def initialize(self, context: PluginContext) -> None:
        """Called by PluginManager when registering the plugin."""
        self.context = context
        try:
            self.on_load()
            logger.info(f"Plugin '{self.name}' v{self.version} initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize plugin '{self.name}': {e}", exc_info=True)
            self.enabled = False

    def on_load(self) -> None:
        """Lifecycle Hook: Executed when the plugin is first loaded into memory."""
        pass

    def on_unload(self) -> None:
        """Lifecycle Hook: Executed when the server shuts down or plugin is removed."""
        pass

    def on_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Lifecycle Hook: Intercepts incoming API requests before scheduling.

        Args:
            request: Mutable dictionary containing request payload (prompt, params, role).

        Returns:
            Modified request dictionary.
        """
        return request

    def on_token_generate(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """Lifecycle Hook: Called during continuous token generation streaming.

        Args:
            token_data: Dict containing 'token_id', 'text', 'logit', 'position'.

        Returns:
            Modified token_data dictionary or filtered payload.
        """
        return token_data

    def on_agent_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Lifecycle Hook: Intercepts ReAct swarm agent steps.

        Args:
            step_data: Dict containing 'agent_role', 'thought', 'tool_name', 'tool_args'.

        Returns:
            Modified step_data dictionary.
        """
        return step_data

    def on_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Lifecycle Hook: Intercepts completed response payloads before client transmission."""
        return response
```

---

## 3. Production Plugin Examples

### Example 1: Enterprise PII & Toxic Guardrail Plugin (`GuardrailPlugin`)

This plugin inspects incoming user prompts for personally identifiable information (PII) using regular expressions and sanitizes matching text before submission to `AsyncPravahaEngine`.

```python
"""Enterprise PII Guardrail Plugin for Pravāha v3.3."""

import re
from typing import Dict, Any
from pravaha.plugins.base_plugin import BasePlugin


class PIIFilterPlugin(BasePlugin):
    name = "pii-filter-guardrail"
    version = "3.3.0"
    description = "Sanitizes SSNs, Credit Cards, and API Keys from prompts."
    priority = 10  # High priority execution

    def on_load(self) -> None:
        self.patterns = {
            "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "CREDIT_CARD": re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
            "API_KEY": re.compile(r"(?i)(?:api_key|access_token|secret)[\s=:\"]+([a-zA-Z0-9_\-]{20,})"),
        }
        print(f"[{self.name}] Compiled {len(self.patterns)} PII redaction patterns.")

    def on_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        messages = request.get("messages", [])
        if not messages:
            return request

        sanitized_count = 0
        for msg in messages:
            if "content" in msg and isinstance(msg["content"], str):
                text = msg["content"]
                for pii_type, regex in self.patterns.items():
                    if regex.search(text):
                        text = regex.sub(f"[REDACTED_{pii_type}]", text)
                        sanitized_count += 1
                msg["content"] = text

        if sanitized_count > 0 and self.context and self.context.audit_trail:
            self.context.audit_trail.log_event(
                event_type="PII_REDACTION_APPLIED",
                details={"redact_count": sanitized_count, "request_id": request.get("request_id")},
            )

        return request
```

---

### Example 2: Custom Agent Tool Registration Plugin (`CustomToolPlugin`)

This plugin registers a custom database query tool (`sql_query_tool`) into Pravāha's `ToolRegistry`, allowing ReAct agents to execute safe SQL queries against an enterprise database.

```python
"""Database Query Tool Plugin for Pravāha ReAct Swarm."""

import sqlite3
from typing import Dict, Any
from pravaha.plugins.base_plugin import BasePlugin


def execute_sql_query(query: str, db_path: str = "./data/enterprise.db") -> str:
    """Safely executes a SELECT query against the enterprise database."""
    if not query.strip().lower().startswith("select"):
        return "Error: Permission Denied. Only read-only SELECT queries are permitted."

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        
        if not rows:
            return "Query returned 0 rows."
            
        result = [dict(zip(columns, row)) for row in rows[:10]] # Limit to 10 rows
        return str(result)
    except Exception as e:
        return f"Database Error: {str(e)}"


class DatabaseToolPlugin(BasePlugin):
    name = "database-tool-extension"
    version = "1.2.0"
    description = "Injects enterprise database querying capability into ToolRegistry."

    def on_load(self) -> None:
        if self.context and self.context.tool_registry:
            # Register tool into the swarm ToolRegistry
            self.context.tool_registry.register_tool(
                name="sql_query_tool",
                func=execute_sql_query,
                description="Executes a read-only SQL query against the enterprise database. Input: SQL string.",
                roles_allowed=["admin", "operator"], # Enforce RBAC scoping
            )
            print(f"[{self.name}] Tool 'sql_query_tool' registered with ToolRegistry.")

    def on_unload(self) -> None:
        if self.context and self.context.tool_registry:
            self.context.tool_registry.unregister_tool("sql_query_tool")
            print(f"[{self.name}] Unregistered 'sql_query_tool'.")
```

---

### Example 3: Cryptographic Audit Ledger Enhancement Plugin (`AuditEnhancerPlugin`)

This plugin hooks into `on_response` to sign every completed response with a SHA-256 HMAC digest before writing the log entry to `SHA256AuditTrail`.

```python
"""Cryptographic Audit Ledger Enhancement Plugin."""

import hmac
import hashlib
import json
from typing import Dict, Any
from pravaha.plugins.base_plugin import BasePlugin


class CryptographicSignerPlugin(BasePlugin):
    name = "crypto-audit-signer"
    version = "2.1.0"
    description = "Appends SHA-256 HMAC signatures to responses and audit trail entries."

    def on_load(self) -> None:
        self.secret_key = b"pravaha-hmac-audit-signing-secret"

    def on_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        response_bytes = json.dumps(response, sort_keys=True).encode("utf-8")
        signature = hmac.new(self.secret_key, response_bytes, hashlib.sha256).hexdigest()

        # Inject signature into response metadata
        if "metadata" not in response:
            response["metadata"] = {}
        response["metadata"]["sha256_signature"] = signature

        # Write to SHA-256 Audit Trail
        if self.context and self.context.audit_trail:
            self.context.audit_trail.log_event(
                event_type="RESPONSE_HMAC_SIGNED",
                details={
                    "request_id": response.get("id"),
                    "signature": signature,
                },
            )

        return response
```

---

## 4. Packaging & Registering Plugins

Pravāha uses standard Python entry points for plugin discovery.

### Packaging via `pyproject.toml`

To create a distributable plugin package:

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "pravaha-pii-guardrail"
version = "1.0.0"
description = "Enterprise PII Filter Plugin for Pravāha"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pravaha>=3.3.0",
]

[project.entry-points."pravaha.plugins"]
pii_filter = "pravaha_pii_guardrail.plugin:PIIFilterPlugin"
```

### Installation Procedures

```bash
# Install local plugin in editable mode
pip install -e ./my_custom_plugin/

# Install plugin directly from Git repository
pip install git+https://github.com/enterprise/pravaha-custom-plugin.git
```

### Configuring Active Plugins in `configs/default.yaml`

```yaml
plugins:
  enabled: true
  auto_discover: true
  active_plugins:
    - pii_filter
    - database-tool-extension
    - crypto-audit-signer
  plugin_configs:
    pii_filter:
      redact_ssn: true
      redact_credit_card: true
    database-tool-extension:
      db_path: "./data/production.db"
```

---

## 5. Plugin Management CLI (`pravaha plugin`)

Pravāha provides CLI commands to manage and audit installed plugins.

```bash
# List all discovered and active plugins
pravaha plugin list

# Inspect detailed metadata and hooks for a specific plugin
pravaha plugin info pii-filter-guardrail

# Dynamically enable or disable a plugin without restarting server
pravaha plugin enable database-tool-extension
pravaha plugin disable database-tool-extension

# Verify plugin compatibility with active Pravāha v3.3 engine
pravaha plugin verify ./my_custom_plugin/
```

---

## 6. Testing Plugins with Pytest

Validate custom plugins using Pravāha's `PluginTestRunner` test utility.

```python
"""Unit tests for PIIFilterPlugin."""

import pytest
from pravaha.plugins.base_plugin import PluginContext
from pravaha_pii_guardrail.plugin import PIIFilterPlugin


@pytest.fixture
def mock_plugin_context():
    return PluginContext(
        engine=None,
        tool_registry=None,
        memory_store=None,
        audit_trail=None,
        config={},
    )


def test_pii_redaction(mock_plugin_context):
    plugin = PIIFilterPlugin()
    plugin.initialize(mock_plugin_context)

    request = {
        "messages": [
            {"role": "user", "content": "My SSN is 123-45-6789 and card is 4111-2222-3333-4444."}
        ]
    }

    result = plugin.on_request(request)
    content = result["messages"][0]["content"]

    assert "123-45-6789" not in content
    assert "[REDACTED_SSN]" in content
    assert "4111-2222-3333-4444" not in content
    assert "[REDACTED_CREDIT_CARD]" in content
```

Execute tests:

```bash
pytest tests/test_custom_plugin.py -v
```

---

## 7. Performance & Security Best Practices

1. **Non-Blocking Execution**: `on_token_generate` is called on the hot path for every token. Never perform blocking disk I/O or network requests inside `on_token_generate`. Offload heavy tasks to background async queues.
2. **Memory Safety**: Do not mutate shared global state inside plugin methods. Use `PluginContext` or thread-safe primitives.
3. **Sandbox Enforcement**: Always declare tool permissions (`roles_allowed`) when registering custom tools into `ToolRegistry`. Any shell or Python tool must delegate execution to `DockerSandbox`.
4. **Exception Handling**: Wrap hook logic in `try/except` blocks so that a failure in a custom plugin does not cause request termination.

# Pravāha v3.3 Developer & Contributor Guide

Thank you for contributing to **Pravāha v3.3**! Pravāha is an enterprise-grade, self-healing LLM inference framework and autonomous agent swarm engine. We maintain rigorous Staff Engineer development standards to ensure numerical stability, memory safety, low latency, and security.

---

## 1. Engineering Principles & Code Philosophy

All contributions must adhere to our core engineering principles:

1. **Defensible Technical Claims**: Never use exaggerated or unsupported marketing statements (e.g., "100% unbreakable", "zero bugs"). Use defensible, empirical language (e.g., "verified within internal benchmark parameters", "bounded latency distribution", "statistically measured").
2. **No Superficial Symptom Patches**: Never resolve bugs by masking symptoms, swallowing exceptions, returning dummy fallbacks, or deleting failing unit tests. Always trace failures upstream to fix the root contractual violation.
3. **Strict Control Flow & Type Safety**: Every public class and function must include complete Python 3.11+ type annotations (`mypy --strict`) and detailed Google-style docstrings.
4. **Empirical Log Verification**: Never declare success after modifying code until you have executed tests and verified clean execution using concrete test runner output.

---

## 2. Development Setup & Workspace Initialization

### 2.1 Prerequisites
Ensure your local environment meets the following software requirements:
- **Python**: Python 3.11 or higher
- **Rust Toolchain**: `rustc` and `cargo` 1.75+ (required for the `pravaha_rust` performance core)
- **C/C++ Compiler**: GCC/Clang or MSVC (for building native dependencies)
- **Docker Engine**: Docker 24.0+ (required for running containerized agent tools)
- **Git**: Git 2.30+

### 2.2 Local Workspace Setup Instructions

```bash
# Step 1: Clone the repository
git clone https://github.com/pravaha/pravaha.git
cd pravaha

# Step 2: Create and activate a Python 3.11 virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# Step 3: Upgrade pip, setuptools, and wheel
python -m pip install --upgrade pip setuptools wheel

# Step 4: Install Pravāha in editable mode with development & test dependencies
pip install -e ".[dev]"

# Step 5: Build the Rust performance extension (PyO3 bindings)
cd rust
cargo build --release
maturin develop --release
cd ..
```

---

## 3. Repository Architecture & Directory Structure Walkthrough

```
pravaha/
├── cache/                # Semantic caching and deduplication engine
│   ├── semantic_dedup.py # SHA-256 and vector-based query deduplication
│   └── cache_manager.py  # Cache replacement algorithms (LRU, LFU)
├── engine/               # Core execution and scheduling engine
│   ├── async_engine.py   # AsyncPravahaEngine entry point and request queueing
│   ├── scheduler.py      # Continuous batching scheduler loop (daemon thread)
│   ├── circuit_breaker.py# CircuitBreaker fault isolation pattern
│   └── model_loader.py   # Dynamic PyTorch / HuggingFace model loader
├── memory/               # KV-Cache and PagedAttention implementation
│   ├── block_manager.py  # Physical KV-block manager and ref counting
│   ├── paged_kv_cache.py # PagedAttention matrix partition mapper
│   └── prefix_cache.py   # Prefix sharing manager
├── observability/        # Observability, telemetry, and security logging
│   ├── audit_trail.py    # Tamper-resistant SHA-256 hash-chained audit log
│   ├── structured_logger.py # JSON logger with X-Request-ID propagation
│   ├── prometheus.py     # Prometheus metric exporters
│   └── pii_filter.py     # Regex and entropy PII redaction filter
├── serving/              # FastAPI application server and routes
│   ├── app.py            # FastAPI initialization and middleware binding
│   ├── middleware.py     # RequestID, Timing, Error, RateLimit, BearerAuth
│   ├── rbac.py           # Role-Based Access Control (ADMIN > OPERATOR > USER)
│   └── routes/           # 12 route modules (completions, chat, swarm, rag, etc.)
├── swarm/                # 52 ReAct autonomous agent swarm
│   ├── base_agent.py     # BaseAgent class with ReAct loop execution
│   ├── orchestrator.py   # Swarm Orchestrator and self-healing loop manager
│   ├── tool_registry.py  # Central tool registry with RBAC checks
│   ├── tools/            # 13 real tools (docker_sandbox, execute_python, etc.)
│   └── agents/           # 52 individual agent implementations
├── tui/                  # Textual terminal user interface
│   ├── dashboard.py      # Main 9-panel dashboard layout
│   └── avatar.py         # Animated robotic ASCII avatar (5 states)
└── cli/                  # Typer + Rich command-line interface
    ├── main.py           # Entry point and command dispatcher
    └── commands/         # Sub-command implementations
rust/                     # High-performance Rust core
├── Cargo.toml            # Rust workspace manifest
└── src/
    ├── lib.rs            # PyO3 module bindings definition
    ├── allocator.rs      # O(1) block allocator with LRU tracking
    ├── prefix_trie.rs    # Token-level prefix trie for O(k) prefix matching
    └── stats.rs          # Real-time memory allocation statistics
```

---

## 4. Coding Standards & Style Conventions

### 4.1 Python Code Formatting & Type Safety
- **Formatting**: Code formatting is enforced using **Ruff**. Line length limit is set to **100 characters**.
- **Type Annotations**: Type hints are strictly required on all function arguments and return types. Python 3.11 `from __future__ import annotations` syntax must be present at the top of every python source file.
- **Docstrings**: Public classes, methods, and functions must contain Google-style docstrings.

```python
from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

class PagedKVCache:
    """Manages physical Key-Value memory page allocations for continuous batching.

    Attributes:
        block_size: Number of token states stored per block.
        num_blocks: Total number of physical memory blocks allocated on target device.
    """

    def __init__(self, block_size: int = 16, num_blocks: int = 1024) -> None:
        self.block_size = block_size
        self.num_blocks = num_blocks
        self._free_blocks: List[int] = list(range(num_blocks))

    def allocate_block(self) -> Optional[int]:
        """Allocates a single free KV block from the available pool.

        Returns:
            The allocated block ID, or None if no free blocks are available.
        """
        if not self._free_blocks:
            logger.warning("KV-cache pool exhausted: zero free blocks remaining.")
            return None
        return self._free_blocks.pop(0)
```

---

### 4.2 Agent Development Rules
When contributing a new swarm agent to `pravaha/swarm/agents/`:
1. **One Class Per File**: Define exactly one agent class per file.
2. **Inherit from `BaseAgent`**: Must subclass `BaseAgent` and call `super().__init__()` with `role`, `priority`, `system_prompt`, and default `available_tools`.
3. **Implement ReAct Loops**: Never substitute agent logic with hardcoded prompt wrapping; use `run_react()`.

```python
# File: pravaha/swarm/agents/my_custom_agent.py
from __future__ import annotations

from pravaha.swarm.base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    """Custom ReAct agent for analyzing system performance metrics."""

    def __init__(self) -> None:
        super().__init__(
            role="custom_performance_analyst",
            priority=5,
            max_tokens=512,
            temperature=0.2,
            system_prompt=(
                "You are an expert Performance Analyst. Analyze throughput and "
                "latency metrics using available tools."
            ),
            available_tools=["execute_python", "read_file"],
            max_react_steps=5,
        )
```

---

### 4.3 Tool Development & Security Isolation
When contributing a new tool to `pravaha/swarm/tools/`:
1. Subclass `BaseTool` in `pravaha/swarm/tools/base_tool.py`.
2. Wrap execution in `DockerSandbox` or `CircuitBreaker` to guarantee containerized isolation and prevent system compromise.
3. Register the tool in `ToolRegistry` with explicit RBAC role requirements (`ADMIN`, `OPERATOR`, `USER`).

```python
# File: pravaha/swarm/tools/custom_sandbox_tool.py
from __future__ import annotations

from typing import Any, Dict
from pravaha.swarm.tools.base_tool import BaseTool
from pravaha.swarm.tools.docker_sandbox import DockerSandbox
from pravaha.engine.circuit_breaker import CircuitBreaker

class CustomSandboxTool(BaseTool):
    """Executes code securely inside an isolated Docker sandbox container."""

    def __init__(self) -> None:
        super().__init__(
            name="custom_sandbox_exec",
            description="Runs code snippet in an isolated sandbox environment.",
            required_role="operator"
        )
        self.sandbox = DockerSandbox(cpu_limit="1.0", memory_limit="256MB")
        self.circuit_breaker = CircuitBreaker(name="custom_sandbox_exec", failure_threshold=3)

    async def execute(self, code: str, timeout: float = 10.0) -> Dict[str, Any]:
        """Execute code securely inside container."""
        def _run():
            return self.sandbox.run_code(code=code, timeout=timeout)

        return await self.circuit_breaker.call_async(_run)
```

---

## 5. Security & Vulnerability Agent Guidelines

Pravāha includes 10 dedicated security agents in `pravaha/swarm/agents/` that map vulnerabilities against the **Common Weakness Enumeration (CWE)** and calculate **CVSS 3.1** metrics.

### Security Agent Checklist
When modifying or extending security auditing agents:
- Ensure all detection rules use static analysis AST pattern matching first, falling back to LLM analysis for contextual validation.
- Every identified vulnerability must report:
  - `cwe_id`: Official CWE identifier (e.g., `CWE-89` for SQL Injection, `CWE-79` for XSS).
  - `cvss_score`: Calculated base CVSS score between `0.0` and `10.0`.
  - `remediation`: Executable fix suggestion.

```python
# Example: CWE Mapping Structure in Security Scanner
VULNERABILITY_CATALOG = {
    "sql_injection": {"cwe_id": "CWE-89", "cvss": 8.8, "severity": "HIGH"},
    "command_injection": {"cwe_id": "CWE-78", "cvss": 9.8, "severity": "CRITICAL"},
    "insecure_deserialization": {"cwe_id": "CWE-502", "cvss": 8.1, "severity": "HIGH"},
    "hardcoded_secrets": {"cwe_id": "CWE-798", "cvss": 7.4, "severity": "HIGH"},
}
```

---

## 6. Rust Performance Core Development Rules

The Rust core (`rust/src/`) provides acceleration for critical memory and search operations.

### Key Rust Guidelines
- **PyO3 Annotations**: Use `#[pyclass]` and `#[pymethods]` to expose Rust structures to Python.
- **Concurrency**: Shared memory structures must use `Arc<RwLock<T>>` or `Arc<Mutex<T>>` for thread-safe access during continuous batching.
- **Zero-Copy**: Return zero-copy references or scalar indices wherever possible to avoid Python GIL overhead.

```rust
// File: rust/src/prefix_trie.rs
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::RwLock;

#[pyclass]
pub struct PrefixTrie {
    root: RwLock<TrieNode>,
}

struct TrieNode {
    children: HashMap<u32, TrieNode>,
    block_id: Option<usize>,
}

#[pymethods]
impl PrefixTrie {
    #[new]
    pub fn new() -> Self {
        PrefixTrie {
            root: RwLock::new(TrieNode {
                children: HashMap::new(),
                block_id: None,
            }),
        }
    }

    pub fn insert(&self, tokens: Vec<u32>, block_id: usize) -> PyResult<()> {
        let mut node = self.root.write().unwrap();
        // Insert tokens into trie...
        Ok(())
    }
}
```

---

## 7. Testing Architecture & Execution

Pravāha features a comprehensive test suite in `tests/`.

### 7.1 Running the Linter & Static Analysis

```bash
# Run Ruff linter check
ruff check pravaha/ tests/

# Run Mypy static type checker
mypy pravaha/ --ignore-missing-imports
```

### 7.2 Running Unit & Integration Tests

```bash
# Run entire test suite with verbose output and coverage reporting
pytest tests/ -v --cov=pravaha

# Run specific phase test suites
pytest tests/test_phase1_engine.py -v
pytest tests/test_phase3_reliability.py -v
pytest tests/test_phase4_hardening.py -v
```

---

## 8. Pull Request & Code Review Process

### 8.1 Branch Naming Conventions
- Features: `feature/short-description` (e.g., `feature/paged-kv-swapping`)
- Bug fixes: `bugfix/issue-number-description` (e.g., `bugfix/fix-race-condition-in-scheduler`)
- Performance: `perf/module-name` (e.g., `perf/rust-trie-lock-contention`)

### 8.2 Pull Request Submission Checklist
Before submitting a PR for review, complete the following verification steps:

- [ ] Run `ruff check pravaha/` and resolve all linting errors.
- [ ] Run `mypy pravaha/` and verify zero type errors.
- [ ] Execute `pytest tests/ -v` and confirm **100% test pass rate**.
- [ ] Ensure all new classes and methods include docstrings and type hints.
- [ ] Verify that audit trail log integrity passes via `AuditTrail.verify_integrity()`.

---

## 9. Staff Engineer Code Review Guidelines

Reviewers verify code quality using the following non-negotiable review criteria:

| Review Dimension | Verification Requirement |
| :--- | :--- |
| **API Preservation** | Function signature changes must update all downstream caller invocations. |
| **Thread Safety** | Inter-thread operations must be protected by explicit locks (`threading.Lock` / `asyncio.Lock`). |
| **Resource Leaks** | Allocations in `BlockManager` must be decremented or freed in `finally:` blocks. |
| **Log Transparency** | Background errors must be logged with complete exception tracebacks using `logger.error(..., exc_info=True)`. |
| **No Dummy Fallbacks**| Failing calls must fail explicitly or trigger circuit breakers rather than silently returning fake values. |

---

## 10. Continuous Integration & Automated Pipelines

Pravāha enforces automated CI checks on every pull request submitted to `main`.

### GitHub Actions Workflow Overview (`.github/workflows/ci.yml`)
1. **Linting & Formatting Stage**:
   - `ruff check pravaha/ tests/`
   - `black --check pravaha/ tests/`
2. **Type Checking Stage**:
   - `mypy pravaha/ --strict`
3. **Rust Extension Compilation Stage**:
   - `cargo test --manifest-path rust/Cargo.toml`
   - `maturin build --release`
4. **Python Test Suite & Coverage Stage**:
   - `pytest tests/ --cov=pravaha --cov-report=xml`
5. **Security Audit Stage**:
   - `bandit -r pravaha/`
   - SHA-256 audit log integrity test pass validation.

---

## 11. Release Engineering & Versioning Policy

Pravāha strictly follows **Semantic Versioning 2.0.0 (MAJOR.MINOR.PATCH)**.

- **MAJOR**: Breaking changes to API routes, CLI command options, or configuration YAML schemas.
- **MINOR**: Backward-compatible new features, new agent swarm capabilities, or performance optimizations in Rust core.
- **PATCH**: Backward-compatible bug fixes and stability improvements.

### Release Workflow
1. Update version strings across `pravaha/__init__.py`, `pyproject.toml`, and `rust/Cargo.toml`.
2. Generate evidence dossier and benchmark validation report.
3. Tag release commit: `git tag -a v3.3.0 -m "Pravaha v3.3.0 Release"`.
4. Push tag to GitHub: `git push origin v3.3.0`.

---

## 12. Troubleshooting & Debugging Cookbook

### 12.1 Debugging Scheduler Deadlocks
If the continuous scheduler thread appears unresponsive:
1. Pass `--verbose` to `pravaha serve` to activate trace logging.
2. Inspect log outputs for `_ready_event` and `_shutdown_event` signals.

### 12.2 Inspecting Tamper-Evident Audit Trail Breaks
If `verify_integrity()` fails:
```python
from pravaha.observability.audit_trail import AuditTrail

audit = AuditTrail("audit_trail.jsonl")
is_valid, msg = audit.verify_integrity()
print(f"Status: {is_valid}, Details: {msg}")
```

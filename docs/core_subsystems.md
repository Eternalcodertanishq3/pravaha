## Core Subsystems Breakdown


### 1. Inference Engine & Continuous Scheduler

Pravāha's engine ([pravaha/engine/async_engine.py](file:///c:/Personal%20Projects/Pravāha/pravaha/engine/async_engine.py)) and continuous scheduler ([pravaha/scheduler/continuous_scheduler.py](file:///c:/Personal%20Projects/Pravāha/pravaha/scheduler/continuous_scheduler.py)) implement iteration-level scheduling:

- **Disjoint Prefill vs. Decode Executions:** To optimize GPU Tensor Core utilization, the scheduler groups requests into either pure prefill batches or pure decode batches during each execution step.
- **Bounded Request Queues:** Unbounded queue growth is prevented by enforcing strict upper bounds:
  - `waiting`: Bounded `collections.deque(maxlen=1000)`
  - `swapped`: Bounded `collections.deque(maxlen=500)`
  - `finished`: Bounded `collections.deque(maxlen=1000)`
  - Streaming Token Queue: `asyncio.Queue(maxsize=200)`
- **Load Shedding & Overload Protection:** The scheduler monitors KV block utilization and queue length via `is_overloaded(0.95)`. When capacity exceeds 95%, incoming requests are shed cleanly with HTTP 429 (Rate Limit) or HTTP 503 (Service Unavailable) responses.
- **Client Disconnect Cleanup:** When a streaming HTTP connection closes prematurely, `async_engine.py` invokes `self.abort_request(request_id)` inside a `finally:` block, freeing allocated KV blocks immediately.

```python
# Engine Initialization & Streaming Inference
from pravaha.config.engine_config import EngineConfig
from pravaha.engine.async_engine import AsyncPravahaEngine
from pravaha.decoder.sampling import SamplingParams

config = EngineConfig.default()
engine = AsyncPravahaEngine(config=config)

# Stream generation with backpressure protection
params = SamplingParams(max_new_tokens=50, temperature=0.7)
async for token in engine.generate("Explain continuous batching", params):
    print(token, end="", flush=True)
```

---

### 2. PagedAttention & Session KV-Cache Manager

Key-Value (KV) cache memory is the primary memory bottleneck in high-concurrency LLM serving. Pravāha addresses this with a two-tier KV cache system:

1. **PagedAttention Block Allocator ([rust/src/allocator.rs](file:///c:/Personal%20Projects/Pravāha/rust/src/allocator.rs)):**
   - Divides KV cache memory into fixed-size physical blocks (e.g., 16 tokens per block).
   - Rust-accelerated `PrefixTrie` tracks shared token prefixes across prompts, enabling zero-copy KV block sharing.
   - Includes Python fallback allocator when compiled Rust binaries are not present.

2. **Session KV-Cache Manager ([pravaha/memory/session_cache.py](file:///c:/Personal%20Projects/Pravāha/pravaha/memory/session_cache.py)):**
   - Maintains stateful multi-turn session histories with LRU eviction.
   - Enforces strict context bounds (`max_context_len=32768`) to prevent out-of-memory errors on long conversations.
   - Configurable Time-To-Live (`ttl_seconds=3600`) automatically purges inactive session allocations.

---

### 3. Swarm Multi-Agent DAG Orchestrator

Pravāha includes a multi-agent orchestration framework ([pravaha/swarm/](file:///c:/Personal%20Projects/Pravāha/pravaha/swarm/)) for complex, multi-step workflows:

- **PipelineDAG (`pravaha/swarm/pipeline_dag.py`):** Enforces directed acyclic graph execution for multi-agent workflows. Detects cycles at construction time and executes agents in topological order.
- **ReActAgent (`pravaha/swarm/agents/react.py`):** Combines reasoning and tool execution in a structured loop. Enforces step bounds (`max_steps=10`) and retry limits to prevent infinite recursion.
- **SharedContext (`pravaha/swarm/memory/`):** Provides thread-safe, locked state sharing across concurrent agent executions.
- **Self-Healing Pipeline (`pravaha/swarm/self_healing.py`):** Automatically intercepts code execution errors, generates fix patches, and re-executes up to `max_repairs=3` iterations before reporting failure.

```python
from pravaha.swarm.pipeline_dag import PipelineDAG
from pravaha.swarm.agents.react import ReActAgent

# Construct multi-agent DAG
dag = PipelineDAG()
researcher = ReActAgent(name="researcher", role="Search internet for data")
coder = ReActAgent(name="coder", role="Write Python processing script")

dag.add_node(researcher)
dag.add_node(coder)
dag.add_edge("researcher", "coder")

# Execute DAG topologically
results = await dag.execute(initial_input="Fetch latest tech news")
```

---

### 4. Security & Sandbox Layer

Security is built directly into Pravāha's serving path across multiple enforcement boundaries:

1. **API Authentication (`pravaha/serving/middleware.py`):** `BearerAuthMiddleware` validates `Authorization: Bearer <key>` headers against `PRAVAHA_API_KEY` for all `/v1/*` and `/admin/*` routes.
2. **Role-Based Access Control (`pravaha/serving/rbac.py`):** Enforces `ADMIN:3` > `OPERATOR:2` > `USER:1` role hierarchy across control routes and tool scopes.
3. **AST Code Execution Sandbox (`pravaha/swarm/tools/python_repl.py`):**
   - Parses code into AST syntax trees before execution via `_validate_ast()`.
   - Blocks dangerous imports (`os`, `subprocess`, `shutil`, `socket`, `http`, `urllib`, `requests`, `httpx`, `ctypes`, `sys`, `pathlib`, `importlib`, `pickle`, `shelve`, `tempfile`).
   - Rejects dangerous calls (`open`, `exec`, `eval`, `compile`, `__import__`).
   - Removes dangerous builtins from the execution namespace.
4. **Docker Tool Sandbox (`pravaha/swarm/tools/docker_sandbox.py`):** Executes external agent code inside isolated Docker containers (`--network none`, `--memory 512m`, `--cpus 1.0`, `--pids-limit 64`).
5. **SSRF Egress Defense (`pravaha/swarm/tools/web_fetcher.py`):** Resolves hostnames via DNS and blocks requests targeting loopback, private, link-local, or cloud metadata IP ranges (`127.0.0.0/8`, `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `169.254.169.254`, `::1`).
6. **Command Injection Prevention (`pravaha/swarm/tools/bash_tool.py`):** Removed `shell=True` entirely. Uses `shlex.split()` and pre-parsing metacharacter checks (`|`, `;`, `&&`, `||`, `$`, `` ` ``) to reject chained command execution.

---

### 5. Data Governance & Privacy Engine

Pravāha enforces privacy compliance and data lifecycle controls:

- **Secrets Redaction Filter ([pravaha/observability/log_filter.py](file:///c:/Personal%20Projects/Pravāha/pravaha/observability/log_filter.py)):** Scans all emitted log messages with regex filters to redact AWS access keys, JWTs, Bearer tokens, private keys, GitHub tokens, and generic password strings before output.
- **PII Redaction Filter ([pravaha/observability/pii_filter.py](file:///c:/Personal%20Projects/Pravāha/pravaha/observability/pii_filter.py)):** Replaces email addresses, US phone numbers, Social Security Numbers, credit card numbers, and IPv4 addresses with redaction tokens (`[EMAIL_REDACTED]`, `[SSN_REDACTED]`).
- **GDPR Compliance APIs ([pravaha/serving/routes/admin.py](file:///c:/Personal%20Projects/Pravāha/pravaha/serving/routes/admin.py)):** Provides authenticated endpoints for user data portability and right-to-be-forgotten requests:
  - `POST /admin/export_user_data`: Exports all stored session KV data and audit traces for a user ID.
  - `POST /admin/delete_user`: Purges all active sessions, cached states, and user records.

---

### 6. Reliability, Circuit Breakers & Audit Ledger

To ensure enterprise uptime and auditable operation:

- **Circuit Breakers ([pravaha/engine/circuit_breaker.py](file:///c:/Personal%20Projects/Pravāha/pravaha/engine/circuit_breaker.py)):** Implements a thread-safe state machine (`CLOSED` → `OPEN` → `HALF_OPEN` → `CLOSED`) to isolate upstream or tool dependencies during failure spikes.
- **Cryptographic Audit Ledger ([pravaha/observability/audit_trail.py](file:///c:/Personal%20Projects/Pravāha/pravaha/observability/audit_trail.py)):** Records all security, administrative, and tool execution events into an append-only, SHA-256 hash-chained log (`hash = SHA256(index:timestamp:event:actor:details:prev_hash)`). Includes `verify_integrity()` tamper detection.
- **Structured JSON Logging ([pravaha/observability/structured_logger.py](file:///c:/Personal%20Projects/Pravāha/pravaha/observability/structured_logger.py)):** Emits machine-readable JSON logs with context-variable correlation IDs (`X-Request-ID`) propagated across async task boundaries.
- **Prometheus Alert Rules ([docker/rules.yml](file:///c:/Personal%20Projects/Pravāha/docker/rules.yml)):** Pre-configured Prometheus alert definitions for queue saturation, high TTFT latency, elevated error rates, and GPU memory pressure.

---

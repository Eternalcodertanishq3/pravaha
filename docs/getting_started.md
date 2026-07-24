# Getting Started with Pravāha v3.3

Pravāha v3.3 is an enterprise-grade, high-performance **Self-Healing LLM Inference Engine and Autonomous Agent Swarm**. Engineered for low-latency batch processing, resilient production deployments, and verifiable security governance, Pravāha combines a high-throughput continuous batching inference engine with a 52-agent autonomous ReAct orchestration system.

---

## 1. System Architecture Overview

Pravāha v3.3 is designed around an eight-layer modular architecture to guarantee complete decoupling between inference scheduling, low-level memory allocation, agentic decision loops, and enterprise security policies.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 1: Interface & API Gateway                                       │
│ FastAPI (OpenAI-compatible) · WebSockets · CLI (Typer) · Textual TUI    │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 2: Engine Orchestration                                           │
│ AsyncPravahaEngine · EventBus · RequestQueue · PriorityScheduler        │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 3: Continuous Inference Pipeline                                  │
│ Dynamic Tokenizer · continuous Scheduler · Sampler · LogitProcessors    │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 4: Memory & Cache Management                                      │
│ PagedKVCache · BlockManager · Rust PrefixTrie (O(1) Matching)           │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 5: Swarm Intelligence (52 Agents)                                 │
│ ReAct Loops · ToolRegistry (13 Tools) · MemoryStore (SQLite + Vector)   │
│ 21 Workers · 12 Auditors · 10 Security Agents · 9 Design Agents        │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 6: Security & Governance                                          │
│ BearerAuthMiddleware · RBACManager · DockerSandbox · SHA256AuditTrail   │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 7: Extensibility & Guardrails                                     │
│ Plugin Architecture · RAG Vector Stores · Guardrail Processors          │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 8: Rust Performance Core                                          │
│ BlockAllocator · PrefixTrie · Real-time AllocatorStats                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Highlights in v3.3

- **PagedAttention & Continuous Scheduling**: Dynamic memory paging avoids external fragmentation in key-value caches. Tokens are batched continuously at iteration boundaries rather than sequence boundaries.
- **Rust Memory Core**: High-throughput block allocation and token prefix trie lookups compiled via Rust FFI (`maturin`), delivering bounded O(1) allocation overhead.
- **52-Agent ReAct Swarm**: Autonomous reasoning loops (THINK $\rightarrow$ ACT $\rightarrow$ OBSERVE $\rightarrow$ THINK $\rightarrow$ ANSWER) equipped with real sandboxed I/O execution tools and SQLite-backed persistent memory.
- **Enterprise Security Plane**: Native `BearerAuthMiddleware`, role-based access control (`RBACManager`), isolated `DockerSandbox` execution environments, and immutable `SHA256AuditTrail` logging.
- **Resilience & Observability**: Integrated `CircuitBreaker` pattern for failure insulation, structured JSON context logger with `request_id_ctx` propagation, and native Prometheus metrics.

---

## 2. Prerequisites & System Requirements

Before installing Pravāha v3.3, verify that your environment satisfies the following system prerequisites:

### Hardware Requirements

| Configuration Profile | Minimum CPU | Recommended GPU | RAM | Storage |
| :--- | :--- | :--- | :--- | :--- |
| **Development / CPU Only** | 4 Cores (x86_64 or ARM64) | None | 16 GB | 20 GB SSD |
| **Small Swarm (7B-8B Models)** | 8 Cores | NVIDIA RTX 3090 / A10G (24GB VRAM) | 32 GB | 50 GB NVMe |
| **Production Enterprise** | 16+ Cores | NVIDIA A100 / H100 (40GB/80GB VRAM) | 64+ GB | 200+ GB NVMe |

### Software Prerequisites

- **Operating System**: Linux (Ubuntu 22.04 LTS recommended), macOS (13+ Apple Silicon supported), or Windows 11 with WSL2.
- **Python**: Version 3.11 or 3.12.
- **Rust Toolchain**: `cargo` 1.75+ and `rustc` for building the performance FFI extension (`maturin`).
- **CUDA Toolkit** (Optional): CUDA 12.1+ with cuDNN for GPU accelerated execution.
- **Docker** (Optional): Docker Engine 24.0+ & Docker Compose v2 for containerized sandboxing and deployment.

---

## 3. Step-by-Step Installation Guide

Pravāha v3.3 can be installed from source or deployed as containerized images.

### Step 1: Clone Repository and Prepare Virtual Environment

```bash
# Clone the repository
git clone https://github.com/pravaha/pravaha.git
cd pravaha

# Create a clean Python 3.11 virtual environment
python3.11 -m venv .venv

# Activate virtual environment
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# Upgrade build tooling
pip install --upgrade pip setuptools wheel maturin
```

### Step 2: Install Package with Dependencies

Pravāha offers modular installation targets depending on your execution environment:

```bash
# Option A: Full Enterprise Suite (GPU, RAG, Swarm, TUI, Dev tooling)
pip install -e ".[all]"

# Option B: GPU-only inference profile
pip install -e ".[gpu]"

# Option C: Minimal CPU evaluation profile
pip install -e "."
```

### Step 3: Build Rust Native Performance Core

The Rust core handles block memory allocation and prefix trie indexing. Compile release binaries directly into your active virtual environment:

```bash
cd rust
maturin develop --release
cd ..
```

Verify that the Rust module is correctly loaded into Python:

```bash
python -c "import pravaha_rust; print(pravaha_rust.__doc__)"
```

---

## 4. Configuration Architecture & Deep-Dive YAML

Pravāha v3.3 uses hierarchical YAML configuration files. Settings are layered: **Built-in Defaults $\rightarrow$ YAML File $\rightarrow$ Environment Variables $\rightarrow$ CLI Arguments**.

### Primary Configuration (`configs/default.yaml`)

Below is an annotated production configuration template:

```yaml
# ==============================================================================
# Pravāha v3.3 Enterprise Core Configuration
# ==============================================================================

engine:
  model_path: "meta-llama/Llama-3-8B-Instruct"
  tokenizer_path: "meta-llama/Llama-3-8B-Instruct"
  quantization: "4bit"           # Options: none, 8bit, 4bit, awq
  device: "cuda"                 # Options: auto, cuda, cpu
  max_model_len: 8192
  dtype: "float16"

scheduler:
  max_num_seqs: 256
  max_num_batched_tokens: 8192
  continuous_batching: true
  waiting_queue_capacity: 1024
  scheduling_policy: "fcfs"       # Options: fcfs, priority

kv_cache:
  block_size: 16
  gpu_memory_utilization: 0.85
  swap_space_gb: 16.0
  enable_prefix_caching: true
  eviction_policy: "lru"

security:
  enable_auth: true
  api_key_env_var: "PRAVAHA_API_KEY"
  enable_rbac: true
  rbac_default_role: "user"
  docker_sandbox:
    enabled: true
    image: "pravaha/sandbox:latest"
    memory_limit: "512m"
    cpu_quota: 100000            # 1.0 CPU core
    timeout_seconds: 10
  audit_trail:
    enabled: true
    log_dir: "./data/audit"
    sha256_verification: true

swarm:
  enabled: true
  max_react_steps: 8
  self_healing: true
  max_audit_iterations: 3
  memory_db_path: "./data/swarm_memory.db"

rag:
  enabled: true
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  vector_store: "faiss"
  index_path: "./data/rag_index.faiss"
  chunk_size: 512
  chunk_overlap: 64

observability:
  log_level: "INFO"
  json_format: true
  enable_prometheus: true
  metrics_port: 8000
  circuit_breaker:
    failure_threshold: 5
    recovery_time_seconds: 30
```

---

## 5. Quick Start Walkthrough

### 1. Launching the Inference Server with TUI

Start Pravāha serving an 8B quantized model with full swarm support and interactive TUI interface:

```bash
pravaha serve meta-llama/Llama-3-8B-Instruct \
  --config configs/default.yaml \
  --quantize 4bit \
  --swarm \
  --tui
```

The Textual TUI dashboard initializes with live metrics, sequence queue states, ASCII avatar animations, and interactive agent logs.

```
┌─ Pravāha v3.3 Control Center ───────────────────────────────────────────┐
│ Avatar: [WORKING] "Analyzing prompt and dispatching to Security Swarm"  │
├─────────────────────────────────────────────────────────────────────────┤
│ Active Requests: 14  │ KV Cache Usage: 42.8%  │ GPU Temp: 61°C          │
│ Total Tokens: 18,420 │ Throughput: 142.3 t/s  │ Circuit: CLOSED         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2. Consuming OpenAI-Compatible REST API

Pravāha exposes OpenAI-compliant endpoints. You can interact using standard cURL commands or official SDKs.

#### Chat Completion Request (`POST /v1/chat/completions`)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${PRAVAHA_API_KEY}" \
  -H "X-User-Role: operator" \
  -d '{
    "model": "meta-llama/Llama-3-8B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a senior systems engineer."},
      {"role": "user", "content": "Explain how PagedAttention reduces memory fragmentation."}
    ],
    "temperature": 0.3,
    "max_tokens": 512,
    "stream": false
  }'
```

#### Response Structure

```json
{
  "id": "chatcmpl-8f92a10e-9411-4f32",
  "object": "chat.completion",
  "created": 1721812995,
  "model": "meta-llama/Llama-3-8B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "PagedAttention addresses virtual memory fragmentation by allocating KV-cache space in non-contiguous physical blocks..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 28,
    "completion_tokens": 142,
    "total_tokens": 170
  }
}
```

### 3. Python Client Integration

```python
import os
from openai import OpenAI

# Initialize client pointing to local Pravāha instance
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=os.getenv("PRAVAHA_API_KEY", "default-dev-key")
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "Generate a Python script to audit directory permissions."}
    ],
    temperature=0.2,
    extra_headers={"X-User-Role": "admin"}
)

print(response.choices[0].message.content)
```

---

## 6. Security & Governance Setup

Pravāha v3.3 enforces robust security parameters across API endpoints and agent tool executions.

### Bearer Authentication Middleware

Configure API authentication by exporting the `PRAVAHA_API_KEY` environment variable:

```bash
export PRAVAHA_API_KEY="sk-pravaha-prod-9a8b7c6d5e4f3a2b"
```

When active, requests lacking a valid `Authorization: Bearer <key>` header are rejected with HTTP 401 Unauthorized, excluding whitelisted health monitoring routes (`/health`, `/health/ready`).

### Role-Based Access Control (RBAC)

RBAC governs access to operational commands and internal agent tools. Roles follow an explicit priority hierarchy:

$$\text{ADMIN (Level 3)} > \text{OPERATOR (Level 2)} > \text{USER (Level 1)}$$

```python
# Example endpoint protection via rbac.py
from fastapi import Depends
from pravaha.serving.rbac import require_role, Role

@app.post("/v1/swarm/execute", dependencies=[Depends(require_role(Role.OPERATOR))])
async def execute_swarm_task(request: SwarmTaskRequest):
    # Only OPERATOR and ADMIN roles can invoke custom agent pipelines
    return await orchestrator.dispatch(request)
```

### Docker Tool Sandbox

Agents executing Python code or shell commands are isolated within a dedicated Docker container environment managed by `pravaha.swarm.tools.docker_sandbox`.

```python
from pravaha.swarm.tools.docker_sandbox import DockerSandbox

sandbox = DockerSandbox(
    image="pravaha/sandbox:latest",
    memory_limit="256m",
    timeout_seconds=5.0
)

result = await sandbox.run_code("print(10 + 20)")
# Returns: {"stdout": "30\n", "exit_code": 0, "timed_out": False}
```

---

## 7. Observability & Self-Diagnostics

### Structured JSON Logging

Pravāha utilizes `StructuredLogger` backed by Python's `contextvars` to correlate log lines across asynchronous tasks using `X-Request-ID`.

```json
{
  "timestamp": "2026-07-24T14:45:10.123Z",
  "level": "INFO",
  "logger": "pravaha.engine.async_engine",
  "request_id": "req-9411-4f32",
  "message": "Continuous batch iteration completed",
  "batched_sequences": 18,
  "allocated_blocks": 142,
  "latency_ms": 12.4
}
```

### Cryptographic Audit Trail

Every state-modifying action (configuration change, agent execution, admin operation) is written to an append-only cryptographic ledger (`SHA256AuditTrail`).

```bash
# Verify integrity of local audit logs
pravaha debug audit-verify --log-dir ./data/audit
```

---

## 8. Development & Continuous Integration Workflows

When extending Pravāha v3.3, execute the full validation test suite to ensure architectural integrity:

```bash
# Run pytest across core engine, swarm, API, and security modules
pytest tests/ -v

# Run type checker
mypy pravaha/

# Run code linter
ruff check pravaha/
```

---

## 9. Common Diagnostic & Resolution Matrix

| Symptom / Error | Probable Root Cause | Resolution Procedure |
| :--- | :--- | :--- |
| `ImportError: cannot import name 'pravaha_rust'` | Rust FFI extension not compiled for active environment. | Run `cd rust && maturin develop --release`. |
| `CUDA out of memory` during startup | `gpu_memory_utilization` threshold set too high. | Lower `gpu_memory_utilization` to `0.70` or enable `--quantize 4bit`. |
| `HTTP 401 Unauthorized` on API endpoints | `PRAVAHA_API_KEY` set on server but missing in request header. | Pass `-H "Authorization: Bearer <key>"` or unset environment variable in dev. |
| `HTTP 403 Forbidden` on Swarm endpoints | Client role (`X-User-Role`) is lower than required role. | Include header `X-User-Role: operator` or `admin`. |
| `DockerSandboxException: Container failed to start` | Docker daemon inactive or user lacks socket permissions. | Verify Docker daemon with `docker ps` and check socket permissions (`/var/run/docker.sock`). |
| `CircuitBreakerOpenException` | Consecutive engine failures exceeded `failure_threshold`. | Inspect logs for upstream exceptions and allow recovery window to elapse (default 30s). |

---

## 10. Summary & Next Steps

Explore detailed technical documentation for deep-dive subsystem topics:

- [Deployment Guide](deployment.md) — Systemd, Docker Compose, Kubernetes, and HA production topologies.
- [Debugging & Profiling Guide](debugging.md) — Token-level introspection, request replay, and logit analysis.
- [Plugin Development Guide](plugins.md) — Custom hooks, middleware extension, and custom agent tools.
- [RAG Pipeline Guide](rag.md) — Vector store integrations (FAISS, Qdrant, PGvector) and hybrid retrieval patterns.

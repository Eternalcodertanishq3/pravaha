# Pravāha v3 — System Architecture

## Overview

Pravāha v3 is a modular, self-healing LLM inference engine built around three core innovations:

1. **Continuous Batching Engine** — PagedAttention with Rust-powered block allocation
2. **32-Agent Swarm** — Specialized agents that collaborate, audit, and self-heal
3. **Full-Stack Serving** — OpenAI-compatible API, TUI dashboard, CLI, and WebSocket streaming

This document describes the internal architecture, data flow, and design decisions.

---

## System Layers

```
┌──────────────────────────────────────────────────────────┐
│  Layer 1: Interface                                      │
│  CLI (Typer) · FastAPI · WebSocket · TUI (Textual)       │
├──────────────────────────────────────────────────────────┤
│  Layer 2: Engine                                         │
│  AsyncPravahaEngine · EventBus · RequestQueue            │
├──────────────────────────────────────────────────────────┤
│  Layer 3: Inference Pipeline                             │
│  Tokenizer → Scheduler → Decoder → Sampler              │
├──────────────────────────────────────────────────────────┤
│  Layer 4: Memory Plane                                   │
│  PagedKVCache · BlockManager · SessionCache              │
│  Prefix Sharing · LRU Swapping · Preemption              │
├──────────────────────────────────────────────────────────┤
│  Layer 5: Intelligence (Swarm)                           │
│  20 Workers · 12 Auditors · Orchestrator · AuditLoop     │
├──────────────────────────────────────────────────────────┤
│  Layer 6: Extensions                                     │
│  RAG · Vision · Branching · Plugins · Guardrails         │
├──────────────────────────────────────────────────────────┤
│  Layer 7: Observability                                  │
│  Prometheus · Tracer · CostEstimator · SelfBenchmark     │
└──────────────────────────────────────────────────────────┘
```

---

## Layer 1: Interface Layer

### CLI (`pravaha/cli/`)

Built with **Typer** + **Rich** for a premium terminal experience.

- `main.py` — Entry point with command registration
- `ascii_art.py` — Banner, gauges, grids, spinners
- `commands/` — 8 command modules: serve, chat, bench, models, swarm, rag, debug, plugins

### FastAPI Server (`pravaha/serving/`)

OpenAI-compatible API server with:

- **11 route modules** under `routes/`
- **3 middleware layers**: RequestID, Timing, ErrorHandler
- **Rate limiting** with in-memory IP tracking
- **WebSocket** endpoint for real-time token streaming
- **CORS** enabled by default

### TUI Dashboard (`pravaha/tui/`)

Built with **Textual** for a full terminal dashboard:

- 8 panels: Header, Chat, Metrics, Queue, Swarm, Audit, RAG, Log
- Dark green terminal aesthetic (`pravaha.tcss`)
- Real-time metrics with ASCII gauge bars

---

## Layer 2: Engine Core

### AsyncPravahaEngine (`pravaha/engine/async_engine.py`)

The central orchestrator. Manages:

1. **Model loading** via `ModelLoader` with dynamic quantization
2. **Background scheduler thread** with `threading.Event` gating (Fix 1)
3. **Request submission** via asyncio Futures
4. **Token streaming** via async generators and cross-thread queues

**Key design decisions:**

- **Thread boundary**: The scheduler runs in a dedicated daemon thread. Tokens cross the thread boundary via `loop.call_soon_threadsafe(queue.put_nowait, token)`.
- **Race condition prevention**: `_ready_event` gates the scheduler loop until all subsystems are initialized.
- **Graceful shutdown**: `_shutdown_event` allows clean termination.

### EventBus (`pravaha/engine/events.py`)

Pub/sub system for telemetry events:

- `MODEL_LOADED` — Fires after model initialization
- `REQUEST_RECEIVED` — New request submitted
- `TOKEN_GENERATED` — First token produced (captures TTFT)
- `REQUEST_COMPLETE` — Request finished (captures throughput)

---

## Layer 3: Inference Pipeline

### Tokenizer (`pravaha/tokenizer/`)

Wraps HuggingFace `AutoTokenizer` with:

- `encode(text)` → token IDs
- `decode_token(id)` → text
- EOS token ID access
- Chat template support

### Continuous Scheduler (`pravaha/scheduler/`)

Implements continuous batching with disjoint execution phases:

1. **Prefill phase**: Batch new requests together for initial token generation
2. **Decode phase**: Batch running requests for subsequent tokens
3. **Preemption**: When slots are full, preempt lowest-priority requests

### Decoder + Sampler (`pravaha/decoder/`)

- `DecoderEngine` — Manages model forward passes with KV cache
- `Sampler` — Temperature scaling → Top-K → Top-P → Repetition penalty → Sample

---

## Layer 4: Memory Plane

### PagedKVCache (`pravaha/memory/paged_cache.py`)

Industrial-grade KV cache with fixed-size 16-token blocks.

### BlockManager (`pravaha/memory/block_manager.py`)

Rust-powered O(1) block allocation with:

- **Prefix sharing**: Multiple requests share physical blocks for identical prefixes
- **LRU swapping**: Least-recently-used blocks swap to CPU RAM under pressure
- **Copy-on-write**: Shared blocks only copy when modified

### SessionKVCache (`pravaha/memory/session_cache.py`)

Multi-turn conversation caching with TTL-based expiry.

---

## Layer 5: Swarm Intelligence

### Agent Architecture

All 32 agents inherit from `BaseAgent` which provides:

```python
class BaseAgent:
    role: str              # Agent identifier
    priority: int          # Execution order (2=orchestrator, 1=senior, 0=worker)
    max_tokens: int        # Token budget
    temperature: float     # Generation temperature
    system_prompt: str     # Expert system prompt

    async def run(task, context, engine) -> AgentOutput
    def can_handle(task_type) -> bool
    async def _generate(prompt, engine) -> str
    async def _generate_json(prompt, engine) -> dict
```

### SharedContext

Cross-agent communication without side channels:

```python
@dataclass
class SharedContext:
    task: str = ""
    output: str = ""
    code: str = ""
    research: str = ""
    reasoning: str = ""
    feedback: str = ""
    plan: str = ""
    agent_outputs: dict[str, AgentOutput]
    audit_reports: list[dict]
    conversation_history: list[dict]
```

### Orchestrator (`pravaha/swarm/orchestrator.py`)

Coordinates agent execution:

1. `execute_agent(name, task, engine)` — Run single agent
2. `execute_pipeline(pipeline, task, engine)` — Run sequence
3. `execute_with_audit(pipeline, task, engine)` — Full audit loop

### AuditLoop (`pravaha/swarm/audit_loop.py`)

The self-healing feedback cycle:

1. Run auditors on output
2. Collect issues from all auditors
3. If issues found → PatchApplier fixes them
4. Re-audit the patched output
5. Repeat up to 3 iterations
6. Return with `AuditResult` (score, issues, patches)

**Output-type-aware auditor selection:**

- **Code**: All 9 auditors (syntax, type, security, logic, etc.)
- **Text**: Logic + Hallucination + Consistency + Verifier
- **Analysis**: Logic + Hallucination + Consistency + EdgeCase + Verifier

### Pipeline Definitions (`pravaha/swarm/pipeline.py`)

6 built-in pipelines:

1. `plan-execute-audit` — General purpose
2. `research-summarize` — Research synthesis
3. `code-review` — Full code quality
4. `creative-write` — Creative content
5. `extract-classify` — Data extraction
6. `reasoning-only` — Pure logic

---

## Layer 6: Extensions

### RAG Pipeline (`pravaha/rag/`)

- Document ingestion (PDF, TXT, MD, HTML, URL)
- Chunking with configurable size/overlap
- Embedding via sentence-transformers
- FAISS vector store
- Top-K retrieval with similarity threshold

### Vision Router (`pravaha/vision/`)

- Image format detection
- Vision model preprocessing
- Multimodal prompt construction (image + text → LLaVA)

### Conversation Branching (`pravaha/branching/`)

- Fork conversations at any message index
- Create labeled branches
- Checkout/delete branches
- Persistent branch store

### Plugin System (`pravaha/plugins/`)

- `BasePlugin` abstract class with lifecycle hooks
- `PluginRegistry` for discovery via entry points
- Hot-loading and unloading

### Guardrails (`pravaha/guardrails/`)

- Content filtering (NSFW, PII, toxicity detection)
- Token budget enforcement (per-request and per-session)

---

## Layer 7: Observability

### Prometheus Metrics (`pravaha/observability/prometheus.py`)

Standard metrics exported at `/metrics`:

- `pravaha_requests_total` — Counter
- `pravaha_tokens_generated` — Counter
- `pravaha_ttft_seconds` — Histogram
- `pravaha_vram_bytes` — Gauge

### Tracer (`pravaha/observability/tracer.py`)

OpenTelemetry-compatible request tracing.

### Cost Estimator (`pravaha/observability/cost_estimator.py`)

Per-request cost estimation based on token counts and model pricing.

### Self-Benchmark (`pravaha/observability/self_benchmark.py`)

Runs on startup to establish baseline throughput and TTFT.

---

## Configuration System

YAML-based configuration with layered defaults:

```
configs/
├── default.yaml          # Full default configuration
├── phase1.yaml           # Minimal CPU testing
├── swarm_default.yaml    # Swarm agent configuration
└── rag_default.yaml      # RAG pipeline configuration
```

Configuration is loaded via `EngineConfig.from_yaml()` with Pydantic validation.

---

## Threading Model

```
Main Thread (asyncio)
│
├─ FastAPI/Uvicorn (async request handling)
│   ├─ engine.generate() → yields tokens via asyncio.Queue
│   └─ WebSocket handler → pushes tokens to client
│
└─ Background Thread (scheduler loop)
    ├─ _ready_event.wait()  ← gates until initialization complete
    ├─ scheduler.step()     ← continuous batching
    ├─ decoder.step_prefill() / step_decode()
    └─ _send_token()        ← loop.call_soon_threadsafe(queue.put_nowait)
```

---

## Security Considerations

1. **Static security scanning**: 8 regex patterns for common vulnerabilities
2. **LLM-based deep analysis**: SecurityAuditAgent scans for OWASP Top 10
3. **Rate limiting**: IP-based middleware with configurable windows
4. **Content filtering**: Guardrail layer for NSFW/PII/toxicity
5. **No eval()**: All agent outputs are parsed, never executed

---

## Testing Strategy

```
tests/
├── test_swarm.py      # Agent registry, SharedContext, routing
├── test_api.py        # Health, models, swarm, middleware
├── test_pipeline.py   # Pipeline validation, agent references
├── test_config.py     # Configuration loading
├── test_decoder.py    # Decoder and sampling
├── test_kv_cache.py   # KV cache operations
├── test_sampling.py   # Sampling strategies
├── test_tokenizer.py  # Tokenizer encode/decode
└── benchmarks/        # Performance benchmarks
```

Run: `pytest tests/ -v --cov=pravaha`

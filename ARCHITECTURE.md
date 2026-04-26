# Pravāha v3.2 — System Architecture

## Overview

Pravāha v3.2 is a **Self-Healing LLM Inference Framework with Autonomous Agent Swarm** built around four core innovations:

1. **ReAct-Based Autonomous Agents** — 51 agents with real tool execution, persistent memory, and self-healing
2. **Continuous Batching Engine** — PagedAttention with Rust-powered block allocation and prefix sharing
3. **Security-First Design** — 10 dedicated security agents with CVSS scoring and CWE mapping
4. **Full-Stack Serving** — OpenAI-compatible API, TUI dashboard with animated avatar, CLI, and WebSocket streaming

---

## System Layers

```
┌──────────────────────────────────────────────────────────┐
│  Layer 1: Interface                                      │
│  CLI (Typer) · FastAPI · WebSocket · TUI (Textual+Avatar)│
├──────────────────────────────────────────────────────────┤
│  Layer 2: Engine                                         │
│  AsyncPravahaEngine · EventBus · RequestQueue            │
├──────────────────────────────────────────────────────────┤
│  Layer 3: Inference Pipeline                             │
│  Tokenizer → Scheduler → Decoder → Sampler              │
├──────────────────────────────────────────────────────────┤
│  Layer 4: Memory Plane                                   │
│  PagedKVCache · BlockManager · SessionCache              │
│  PrefixTrie (Rust) — O(k) prefix matching, integrated as  │
│  primary path with SHA-256 hash fallback                   │
│  LRU Swapping · Preemption                                 │
├──────────────────────────────────────────────────────────┤
│  Layer 5: Intelligence (Swarm — 51 Agents)               │
│  20 Workers · 12 Auditors · 10 Security · 9 Design      │
│  ReAct Loop · ToolRegistry · Persistent Memory           │
├──────────────────────────────────────────────────────────┤
│  Layer 6: Extensions                                     │
│  RAG · Vision · Branching · Plugins · Guardrails         │
├──────────────────────────────────────────────────────────┤
│  Layer 7: Observability                                  │
│  Prometheus · Tracer · CostEstimator · SelfBenchmark     │
├──────────────────────────────────────────────────────────┤
│  Layer 8: Rust Performance Core                          │
│  BlockAllocator · PrefixTrie · AllocatorStats            │
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

- 9 panels: Header, Chat, Metrics, Queue, Swarm, Audit, RAG, Log, **Avatar**
- Pixel-art animated avatar with 5 states (idle, thinking, working, success, audit)
- Dark green terminal aesthetic (`pravaha.tcss`)
- Real-time metrics with ASCII gauge bars

---

## Layer 2: Engine Core

### AsyncPravahaEngine (`pravaha/engine/async_engine.py`)

The central orchestrator. Manages:

1. **Model loading** via `ModelLoader` with dynamic quantization
2. **Background scheduler thread** with `threading.Event` gating
3. **Request submission** via asyncio Futures
4. **Token streaming** via async generators and cross-thread queues

**Key design decisions:**

- **Thread boundary**: The scheduler runs in a dedicated daemon thread. Tokens cross via `loop.call_soon_threadsafe(queue.put_nowait, token)`.
- **Race condition prevention**: `_ready_event` gates the scheduler loop until initialization is complete.
- **Graceful shutdown**: `_shutdown_event` allows clean termination.

---

## Layer 5: Swarm Intelligence (v3.1 — Complete Rewrite)

### The Fundamental Change: ReAct vs Prompt Wrapping

**Before (v3.0):**
```python
prompt = self.build_prompt(task, context)
output = await self._generate(prompt, engine)
return AgentOutput(output=output)  # ← This is NOT agentic
```

**After (v3.1):**
```python
# ReAct loop: THINK → ACT → OBSERVE → THINK → ... → ANSWER
for step in range(self.max_react_steps):
    output = await self._generate(react_prompt, engine)
    parsed = self._parse_react_output(output)
    
    if parsed.is_final_answer:
        return self._build_output(answer=parsed.answer)
    
    if parsed.action and self._tool_registry:
        observation = await self._tool_registry.execute(
            tool_name=parsed.action.tool_name,
            args=parsed.action.args,
        )
        react_prompt += f"\nObservation: {observation}\nThought:"
```

### Agent Architecture

All 51 agents inherit from `BaseAgent` which provides:

```python
class BaseAgent(ABC):
    role: str              # Agent identifier
    priority: int          # Execution order
    max_tokens: int        # Token budget
    temperature: float     # Generation temperature
    system_prompt: str     # Expert system prompt
    available_tools: list  # Tools this agent can use
    max_react_steps: int   # Max ReAct iterations
    
    async def run_react(task, context, engine) -> AgentOutput  # ReAct loop
    async def run(task, context, engine) -> AgentOutput         # Auto-selects
    def attach_tools(registry: ToolRegistry) -> None
    def attach_memory(memory: AgentMemory) -> None
```

### Tool System

Real tools that execute actual I/O operations:

| Tool | Capability | Security |
|------|-----------|----------|
| `execute_python` | Subprocess sandbox (5s timeout, 256MB) | No shell=True, sanitized env |
| `read_file` | Local file reading | Whitelisted extensions only |
| `fetch_url` | HTTP GET + HTML→text | 10s timeout |
| `web_search` | DuckDuckGo API | No API key required |
| `run_shell` | Shell commands | Whitelisted commands/flags only |
| `memory` | Persistent memory | Namespaced per agent role |

### Persistent Memory System

SQLite-backed memory that persists across sessions:

- **MemoryStore** — WAL-mode SQLite with importance weighting, access-time tracking, text search
- **EpisodicMemory** — Task-result episodes for learning from past outcomes
- **SemanticMemory** — TF-IDF cosine similarity for fact retrieval

### Orchestrator (`pravaha/swarm/orchestrator.py`)

Coordinates the 51-agent swarm:

1. Initializes `ToolRegistry` and `MemoryStore` for every agent on construction
2. `execute_agent(name, task, engine)` — Run single agent with avatar state
3. `execute_pipeline(pipeline, task, engine)` — Run sequence
4. `execute_with_audit(pipeline, task, engine)` — Full self-healing audit loop

### Self-Healing Audit Loop

```
Worker Pipeline → Audit Pass → Issues? → PatchApplier → Re-Audit → ...
                                  ↓ No
                              ✅ Output (score ≥ 70)
```

- Runs up to 3 iterations
- 12 auditor agents (regex-first, then LLM)
- PatchApplier auto-fixes issues between iterations
- OutputVerifier gates final release with confidence score

### Security Agents (10)

Dedicated security analysis with CVSS scoring:

| Agent | Focus | Patterns |
|-------|-------|:--------:|
| SecurityAudit | eval/exec/pickle + CWE mapping | 12 |
| InjectionScanner | SQL/XSS/XXE/command injection | 10 |
| AuthAudit | JWT/session/credentials | 5 |
| CryptoAudit | MD5/SHA1/DES/ECB/weak keys | 8 |
| DependencyAudit | Risky imports | 6 |
| SecretsScanner | API keys + Shannon entropy | 8+entropy |
| NetworkSecurity | HTTP/SSL/CORS/SSRF | 5 |
| PrivilegeAudit | Root escalation/chmod 777 | 5 |
| APISecurity | Rate limiting/mass assignment | 4 |
| Compliance | GDPR/PCI/OWASP | 5 |

### Design Agents (9)

UI/UX design with WCAG accessibility auditing:

| Agent | Focus |
|-------|-------|
| UIDesigner | Structured JSON design specs |
| ComponentBuilder | React/HTML/CSS implementation |
| LayoutDesigner | CSS Grid/Flexbox optimization |
| StyleDesigner | Design token systems |
| AccessibilityAuditor | WCAG 2.1 AA (6 regex checks) |
| UXReviewer | Nielsen's 10 heuristics |
| DesignCritic | 5-dimension quality scoring |
| PrototypeBuilder | Single-file HTML prototypes |
| DesignSystem | Token + pattern library architecture |

---

## Layer 8: Rust Performance Core

### BlockAllocator (`rust/src/allocator.rs`)

O(1) block allocation with:

- **allocate/free** — Single block operations with LRU tracking
- **batch_allocate** — Atomic multi-request allocation
- **evict_lru_batch** — Bulk LRU eviction for memory pressure
- **get_block_age** — Block age calculation for eviction decisions

### PrefixTrie (`rust/src/prefix_trie.rs`)

Token-level prefix trie for O(1) average prefix matching:

- Uses `Arc<RwLock<T>>` for concurrent read access during continuous batching
- `insert(tokens, block_id)` — Map token sequence to block
- `longest_prefix_match(tokens)` — Find longest shared prefix
- `decrement_ref(tokens)` — Return freed block IDs

### AllocatorStats (`rust/src/stats.rs`)

Real-time observability:

- `hit_rate()` — Prefix cache hit ratio
- `utilization(num_blocks)` — Block utilization percentage
- `alloc_free_ratio()` — Allocation/free ratio

---

## Configuration System

YAML-based configuration with layered defaults:

```
configs/
├── default.yaml          # Full default configuration
├── phase1.yaml           # Minimal CPU testing
├── swarm_default.yaml    # 51-agent swarm configuration
└── rag_default.yaml      # RAG pipeline configuration
```

---

## Testing Strategy

```
tests/
├── test_swarm.py              # 51-agent registry, SharedContext, routing
├── test_api.py                # Health, models, swarm, middleware
├── test_pipeline.py           # Pipeline validation
├── test_security_agents.py    # Security static scan verification
├── test_design_agents.py      # Accessibility + design agent tests
├── test_react_loop.py         # ReAct loop, tool parsing, sandboxed tools
├── test_agents_runtime.py     # Real agent execution with MockEngine
├── test_memory.py             # MemoryStore, EpisodicMemory, SemanticMemory
├── test_branching_fixed.py    # Branching CRUD operations
├── test_debug_routes.py       # Debug routes + replayer
└── benchmarks/                # Performance benchmarks
```

Run: `pytest tests/ -v` (76 tests, all passing)

---

## Security Considerations

1. **10 dedicated security agents** with CVSS scoring and CWE mapping
2. **Shannon entropy detection** for unknown secret patterns (entropy > 4.5)
3. **Sandboxed code execution** — 5s timeout, 256MB memory, no shell=True
4. **Whitelisted file access** — Only .py, .js, .ts, .md, .json, .yaml, .toml
5. **Whitelisted shell commands** — Blocks rm, sudo, chmod, curl, wget
6. **Rate limiting** — IP-based middleware with configurable windows
7. **Content filtering** — Guardrail layer for NSFW/PII/toxicity
8. **No eval()** — All agent outputs are parsed, never executed

---

## Inference Performance Note

Pravāha's inference path wraps HuggingFace Transformers.
It does not use custom CUDA kernels. For benchmarks, measure
on your actual hardware. Approximate numbers:

| Setup | TTFT |
|-------|------|
| GPT-2 on CPU | ~80ms |
| Llama-3-8B on A100 (4-bit) | ~40-60ms |

For pure inference throughput, vLLM with custom CUDA kernels
achieves 3-5x higher token generation speed. Pravāha's value
is in the intelligent layer: agents, self-healing, RAG, and
observable workflows.

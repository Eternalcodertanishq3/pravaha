# Pravāha v3.1 — प्रवाह

### Full-Stack Autonomous AI Inference Operating System

> 51 agents. ReAct-based autonomy. Self-healing audit pipeline. Persistent memory. Sandboxed tool execution. RAG. Vision routing. Conversation branching. Rust performance core.

[![CI](https://github.com/pravaha/pravaha/actions/workflows/ci.yml/badge.svg)](https://github.com/pravaha/pravaha/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)

---

## Why Pravāha?

Most LLM inference tools solve **one** problem. Pravāha solves **all of them**:

| Capability | vLLM | Ollama | llama.cpp | **Pravāha v3.1** |
|---|:---:|:---:|:---:|:---:|
| Continuous Batching | ✅ | ✅ | ✅ | ✅ |
| PagedAttention | ✅ | ✅ | ✅ | ✅ |
| OpenAI-Compatible API | ✅ | ✅ | ✅ | ✅ |
| 51-Agent Autonomous Swarm | ❌ | ❌ | ❌ | ✅ |
| ReAct Loop (Reason + Act) | ❌ | ❌ | ❌ | ✅ |
| Self-Healing Audit Loop | ❌ | ❌ | ❌ | ✅ |
| Persistent Agent Memory | ❌ | ❌ | ❌ | ✅ |
| Sandboxed Tool Execution | ❌ | ❌ | ❌ | ✅ |
| 10 Security Audit Agents | ❌ | ❌ | ❌ | ✅ |
| 9 Design Agents | ❌ | ❌ | ❌ | ✅ |
| Built-in RAG Pipeline | ❌ | ✅ | ❌ | ✅ |
| Vision Routing | ❌ | ✅ | ❌ | ✅ |
| Conversation Branching | ❌ | ❌ | ❌ | ✅ |
| Terminal Dashboard (TUI) | ❌ | ❌ | ❌ | ✅ |
| Pixel Avatar Animation | ❌ | ❌ | ❌ | ✅ |
| Rust Performance Core | ✅ | ❌ | ✅ | ✅ |
| Plugin System | ❌ | ❌ | ❌ | ✅ |
| Token-Level Debugging | ❌ | ❌ | ❌ | ✅ |

> **Think of it this way:**
> - **LLaMA** = the car engine 🛠️
> - **CUDA** = the fuel ⛽
> - **Pravāha** = the entire self-driving race car 🏎️ with pit crew, dashboard, and autopilot

---

## What Makes v3.1 Different: True Autonomy

Every agent in Pravāha v3.1 uses the **ReAct (Reason + Act) loop**:

```
THINK → ACT → OBSERVE → THINK → ACT → OBSERVE → ... → ANSWER
```

This is NOT prompt wrapping. Agents:
1. **Plan** their own sub-steps before executing
2. **Execute real tools** (code runner, web search, file reader)
3. **Observe** results and adapt
4. **Persist memory** across sessions (SQLite-backed)
5. **Self-heal** through a 12-auditor feedback loop

---

## Quick Start

### One-Command Serving

```bash
# Install
pip install -e ".[all]"

# Serve any HuggingFace model with one command
pravaha serve gpt2
pravaha serve meta-llama/Llama-3-8B --quantize 4bit --tui
pravaha serve mistralai/Mistral-7B --swarm --self-heal --rag --tui

# Interactive chat
pravaha chat --server http://localhost:8000

# Run benchmarks
pravaha bench --model gpt2 --runs 5
```

### Python API

```python
from pravaha.engine.async_engine import AsyncPravahaEngine
from pravaha.config.engine_config import EngineConfig

config = EngineConfig(model_name="gpt2", quantization="4bit")
engine = AsyncPravahaEngine(config=config)

async for token in engine.generate("Explain quantum computing"):
    print(token, end="", flush=True)
```

### Docker

```bash
docker compose -f docker/docker-compose.yml up
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  Layer 1: Interface                                        │
│  CLI (Typer) · FastAPI · WebSocket · TUI (Textual+Avatar) │
├────────────────────────────────────────────────────────────┤
│  Layer 2: Engine                                           │
│  AsyncPravahaEngine · EventBus · RequestQueue              │
├────────────────────────────────────────────────────────────┤
│  Layer 3: Inference Pipeline                               │
│  Tokenizer → Scheduler → Decoder → Sampler                │
├────────────────────────────────────────────────────────────┤
│  Layer 4: Memory Plane                                     │
│  PagedKVCache · BlockManager · PrefixTrie (Rust)           │
│  Prefix Sharing · LRU Swapping · Preemption                │
├────────────────────────────────────────────────────────────┤
│  Layer 5: Intelligence (Swarm — 51 Agents)                │
│  20 Workers · 12 Auditors · 10 Security · 9 Design        │
│  ReAct Loop · Tools · Persistent Memory                    │
├────────────────────────────────────────────────────────────┤
│  Layer 6: Extensions                                       │
│  RAG · Vision · Branching · Plugins · Guardrails           │
├────────────────────────────────────────────────────────────┤
│  Layer 7: Observability                                    │
│  Prometheus · Tracer · CostEstimator · SelfBenchmark       │
├────────────────────────────────────────────────────────────┤
│  Layer 8: Rust Performance Core                            │
│  BlockAllocator · PrefixTrie · AllocatorStats              │
└────────────────────────────────────────────────────────────┘
```

---

## 51-Agent Swarm

### Workers (20 agents)
| Agent | Role | ReAct? | Tools |
|-------|------|:------:|-------|
| PlannerAgent | Task decomposition | ✅ | memory |
| CoderAgent | Code generation + verification | ✅ | execute_python, read_file, web_search |
| DebuggerAgent | Root cause analysis + fix | ✅ | execute_python, read_file |
| ResearcherAgent | Web research + cross-reference | ✅ | web_search, fetch_url |
| ReasoningAgent | Chain-of-thought + math verify | ✅ | execute_python |
| CriticAgent | Quality critique | — | — |
| ValidatorAgent | Output validation | — | — |
| SummarizerAgent | Text summarization | — | — |
| ExpanderAgent | Content expansion | — | — |
| TranslatorAgent | Language translation | — | — |
| MergerAgent | Output merging | — | — |
| RouterAgent | Task routing | — | — |
| MemoryAgent | Memory management | — | — |
| ToolAgent | Tool orchestration | ✅ | all tools |
| JudgeAgent | Quality judging | — | — |
| RefinerAgent | Output refinement | — | — |
| ClassifierAgent | Task classification | — | — |
| ExtractorAgent | Data extraction | — | — |
| NarratorAgent | Narrative writing | — | — |
| EnsembleAgent | Multi-model ensemble | — | — |

### Auditors (12 agents)
Static regex-first analysis (zero LLM cost for detection):

| Agent | Patterns | Focus |
|-------|:--------:|-------|
| SyntaxAuditAgent | 7 | eval, exec, bare except, star import, mutable default, global, assert |
| TypeSafetyAgent | 3 | isinstance chains, bare type(), Any overuse |
| LogicFlawAgent | 4 | == None, while True break, unreachable code, empty catch |
| PerformanceProfilerAgent | 3 | nested loops, string concat, repeated computation |
| ConsistencyGuardAgent | — | Cross-output consistency |
| HallucinationHunterAgent | — | Factual verification |
| EdgeCaseHunterAgent | — | Boundary conditions |
| OutputVerifierAgent | — | Final quality gate |
| PatchApplierAgent | — | Auto-fix issues |
| SelfReflectionAgent | — | Meta-cognitive review |
| TestGeneratorAgent | — | Auto-generate tests |
| RegressionGuardAgent | — | Detect regressions from patches |

### Security Agents (10)
| Agent | Patterns | CVSS? | Focus |
|-------|:--------:|:-----:|-------|
| SecurityAuditAgent | 12 | ✅ | eval/exec/pickle + CWE mapping |
| InjectionScannerAgent | 10 | — | SQL/XSS/XXE/command/template injection |
| AuthAuditAgent | 5 | — | JWT, session fixation, hardcoded creds |
| CryptoAuditAgent | 8 | — | MD5/SHA1/DES/RC4/ECB/weak keys |
| DependencyAuditAgent | 6 | — | pickle/marshal/ctypes/telnet |
| SecretsScannerAgent | 8+entropy | — | AWS/GitHub/OpenAI/Slack + Shannon entropy |
| NetworkSecurityAgent | 5 | — | HTTP/SSL/CORS/bind/SSRF |
| PrivilegeAuditAgent | 5 | — | Root escalation, chmod 777 |
| APISecurityAgent | 4 | — | Rate limiting, header injection |
| ComplianceAgent | 5 | — | GDPR/PCI/OWASP logging |

### Design Agents (9)
| Agent | Role | Tools | Focus |
|-------|------|-------|-------|
| UIDesignerAgent | ui_designer | web_search | Layout + visual + interaction specs |
| ComponentBuilderAgent | component_builder | execute_python | React/HTML/CSS components |
| LayoutAgent | layout_designer | — | CSS Grid/Flexbox layouts |
| StyleAgent | style_designer | — | Design token systems |
| AccessibilityAgent | accessibility_auditor | — | WCAG 2.1 AA compliance (6 checks) |
| UXReviewerAgent | ux_reviewer | — | Nielsen's 10 heuristics |
| DesignCriticAgent | design_critic | — | 5-dimension scoring |
| PrototypeAgent | prototype_builder | read_file | Single-file HTML prototypes |
| DesignSystemAgent | design_system | — | Token + pattern library |

---

## Tool System

Agents can execute **real tools** during the ReAct loop:

| Tool | Name | Description | Security |
|------|------|-------------|----------|
| CodeExecutor | `execute_python` | Subprocess sandbox, 5s timeout | No shell=True, 8KB max output |
| FileReader | `read_file` | Whitelisted extensions only | .py,.js,.ts,.md,.json,.yaml,.toml |
| WebFetcher | `fetch_url` | HTTP GET + HTML→text | 10s timeout, follow redirects |
| SearchTool | `web_search` | DuckDuckGo API | No API key needed |
| ShellRunner | `run_shell` | Whitelisted commands only | Blocked: rm, sudo, chmod, curl |
| MemoryTool | `memory` | Agent-scoped SQLite store | Namespaced per agent role |

---

## Persistent Memory

Agents maintain memory across sessions via SQLite (WAL mode):

| Module | Purpose | Key Feature |
|--------|---------|-------------|
| `MemoryStore` | Key-value store | Importance weighting, access-time tracking |
| `EpisodicMemory` | Task-result episodes | Keyword-overlap recall for learning |
| `SemanticMemory` | Fact store | TF-IDF cosine similarity |

---

## Rust Performance Core

| Module | Thread Safety | Key Methods |
|--------|:------------:|-------------|
| `BlockAllocator` | — | allocate, free, batch_allocate, evict_lru_batch |
| `PrefixTrie` | `Arc<RwLock>` | insert, longest_prefix_match, decrement_ref |
| `AllocatorStats` | — | hit_rate(), utilization(), alloc_free_ratio() |

---

## TUI Dashboard

```
┌──────────────────────────────────────────────────────┐
│  PRAVAHA v3.1  ·  Llama-3  ·  4-bit  ·  RTX4090     │
├──────────┬───────────────────┬────────────────────────┤
│ [AVATAR] │   Chat Panel      │   Metrics Panel        │
│  ╭━━━━━╮ │   (streaming)     │   Throughput gauge      │
│  │ ◉  ◉ ││                   │   VRAM gauge            │
│  │  ━   ││                   │   Queue bar             │
│  ╰━━━━━╯ │                   │                        │
├──────────┴───────────────────┴────────────────────────┤
│  Agents: [plan●][code●][crit○][synx●][halu○]...       │
├──────────────────────────────────────────────────────┤
│  Audit: [SyntaxAudit: scanning...] iter=1 issues=2   │
├──────────────────────────────────────────────────────┤
│  Logs: 14:22:31 INFO Prefill batch=4, time=1.02s     │
└──────────────────────────────────────────────────────┘
```

Avatar states: **idle** (cyan) → **thinking** (yellow) → **working** (green) → **audit** (magenta) → **success** (green ✦)

---

## Pipelines

| Pipeline | Workers | Auditors |
|----------|---------|----------|
| `plan-execute-audit` | planner → coder → critic | syntax + security + verifier |
| `research-summarize` | researcher → reasoning → summarizer | hallucination + consistency |
| `code-review` | coder → debugger → critic → refiner | syntax + type + security + perf + test |
| `secure-code-review` | planner → coder → debugger | ALL 10 security agents |
| `design-component` | ui_designer → layout → style → builder | accessibility + UX + critic |
| `full-secure-design` | planner → designer → builder → coder → debug | security + design + perf |

---

## Configuration

YAML-based with layered defaults:

```
configs/
├── default.yaml          # Full engine configuration
├── phase1.yaml           # Minimal CPU testing
├── swarm_default.yaml    # 51-agent swarm configuration
└── rag_default.yaml      # RAG pipeline configuration
```

---

## Testing

```bash
# Run all tests (76 tests)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pravaha

# Run specific test suites
pytest tests/test_swarm.py           # Agent registry (51 agents)
pytest tests/test_security_agents.py # Security static scans
pytest tests/test_design_agents.py   # Accessibility + design
pytest tests/test_react_loop.py      # ReAct loop + tools
pytest tests/test_memory.py          # Persistent memory
pytest tests/test_agents_runtime.py  # Real agent runtime
```

---

## Project Structure

```
pravaha/
├── cli/                    # CLI commands (Typer + Rich)
├── config/                 # Configuration system (Pydantic)
├── engine/                 # Async inference engine + EventBus
├── scheduler/              # Continuous batching scheduler
├── decoder/                # Model forward pass + sampling
├── memory/                 # PagedKVCache + BlockManager
├── tokenizer/              # HuggingFace tokenizer wrapper
├── serving/                # FastAPI server + 11 routes
├── swarm/
│   ├── agents/
│   │   ├── workers/        # 20 worker agents (5 ReAct-enabled)
│   │   ├── auditors/       # 12 audit agents (regex-first)
│   │   ├── security/       # 10 security agents (CVSS+CWE)
│   │   └── design/         # 9 design agents (WCAG+Nielsen)
│   ├── tools/              # 6 sandboxed tools
│   ├── memory/             # SQLite persistent memory
│   ├── orchestrator.py     # Agent coordination
│   └── pipeline.py         # 8 named pipelines
├── tui/                    # Terminal dashboard + avatar
├── rag/                    # RAG pipeline (FAISS + embeddings)
├── vision/                 # Multimodal vision routing
├── branching/              # Conversation branching
├── debug/                  # Replayer + StepDebugger + Tracer
├── plugins/                # Plugin system
├── guardrails/             # Content filtering
├── observability/          # Prometheus + cost estimation
└── rust/src/               # Rust performance core
    ├── allocator.rs         # Block allocator + batch ops
    ├── prefix_trie.rs       # O(1) prefix matching trie
    ├── stats.rs             # Allocation statistics
    └── lib.rs               # PyO3 module exports
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Pravāha v3.1</b> — The claim of "Full-Stack Autonomous AI Inference OS" is earned.<br>
  Not a swarm of system prompts. A genuine inference operating system.
</p>

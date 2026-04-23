# Pravāha v3 — प्रवाह

### The self-healing, swarm-ready LLM inference engine.

> 32 agents. Self-auditing pipeline. RAG built-in. Vision routing. Conversation branching. Plugin system.

[![CI](https://github.com/pravaha/pravaha/actions/workflows/ci.yml/badge.svg)](https://github.com/pravaha/pravaha/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)

---

## Why Pravāha?

Most LLM inference tools solve **one** problem. Pravāha solves **all of them**:

| Capability | vLLM | Ollama | llama.cpp | **Pravāha v3** |
|---|:---:|:---:|:---:|:---:|
| Continuous Batching | ✅ | ❌ | ❌ | ✅ |
| PagedAttention | ✅ | ❌ | ❌ | ✅ |
| OpenAI-Compatible API | ✅ | ✅ | ✅ | ✅ |
| 32-Agent Swarm | ❌ | ❌ | ❌ | ✅ |
| Self-Healing Audit Loop | ❌ | ❌ | ❌ | ✅ |
| Built-in RAG Pipeline | ❌ | ❌ | ❌ | ✅ |
| Vision Routing | ❌ | ❌ | ❌ | ✅ |
| Conversation Branching | ❌ | ❌ | ❌ | ✅ |
| Terminal Dashboard (TUI) | ❌ | ❌ | ❌ | ✅ |
| Plugin System | ❌ | ❌ | ❌ | ✅ |
| Token-Level Debugging | ❌ | ❌ | ❌ | ✅ |
| Request Replay | ❌ | ❌ | ❌ | ✅ |

> **Think of it this way:**
> - **LLaMA** = the car engine 🛠️
> - **CUDA** = the fuel ⛽
> - **Pravāha** = the entire self-driving race car 🏎️ with pit crew, dashboard, and autopilot

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
from pravaha.decoder.sampling import SamplingParams

engine = AsyncPravahaEngine(config_path="configs/default.yaml")

async for token in engine.generate("Explain quantum computing", SamplingParams()):
    print(token, end="", flush=True)
```

### OpenAI-Compatible API

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="gpt2",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

### Docker

```bash
docker compose -f docker/docker-compose.yml up
# Pravaha on :8000, Prometheus on :9090, Grafana on :3000
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Pravāha v3 Engine                        │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │   CLI    │  │ FastAPI  │  │ WebSocket│  │    TUI    │  │
│  │  Typer   │  │  Server  │  │ Streaming│  │  Textual  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬─────┘  │
│       └──────────────┼───────────────────────────┘          │
│                      ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              AsyncPravahaEngine                      │   │
│  │  ┌──────────┐ ┌────────────┐ ┌───────────────────┐  │   │
│  │  │ Tokenizer│ │   Decoder  │ │    Scheduler      │  │   │
│  │  └──────────┘ │  + Sampler │ │ ContinuousBatch   │  │   │
│  │               └──────┬─────┘ └──────────┬────────┘  │   │
│  │                      ▼                  ▼            │   │
│  │  ┌──────────────────────────────────────────────┐    │   │
│  │  │  Memory Plane (Rust BlockAllocator)          │    │   │
│  │  │  PagedKVCache · BlockManager · SessionCache  │    │   │
│  │  │  Prefix Sharing · LRU Swapping · Preemption  │    │   │
│  │  └──────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Swarm Layer (32 Agents)                 │   │
│  │  ┌─ Workers (20) ──────────────────────────────────┐ │   │
│  │  │ Planner·Coder·Debugger·Critic·Researcher·...   │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  │  ┌─ Auditors (12) ─────────────────────────────────┐ │   │
│  │  │ SyntaxAudit·Security·TypeSafety·EdgeCase·...    │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  │  ┌─ Self-Healing Loop ─────────────────────────────┐ │   │
│  │  │ Audit → Find Issues → Patch → Re-Audit → Pass  │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌────────┐ ┌───────────┐ ┌──────────┐ ┌───────────────┐   │
│  │  RAG   │ │  Vision   │ │Branching │ │   Plugins     │   │
│  │Pipeline│ │  Router   │ │ Manager  │ │   Registry    │   │
│  └────────┘ └───────────┘ └──────────┘ └───────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Observability: Prometheus · Tracer · CostEstimator  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## The 32-Agent Swarm

Pravāha's swarm is a multi-agent system where specialized agents collaborate to produce, audit, and refine outputs.

### 20 Worker Agents

| Agent | Role | Key Feature |
|---|---|---|
| **PlannerAgent** | Task decomposition | Tags subtasks by type + complexity |
| **ResearcherAgent** | Information gathering | Confidence-tagged claims [HIGH/MEDIUM/LOW] |
| **CoderAgent** | Code generation | Language detection, quality metrics |
| **DebuggerAgent** | Root cause analysis | Structured bug reports with minimal fixes |
| **CriticAgent** | Quality evaluation | 4-dimensional scoring (clarity/correctness/completeness/efficiency) |
| **ValidatorAgent** | Fact verification | Per-claim [V]/[?]/[X] tagging |
| **SummarizerAgent** | Content condensation | Compression ratio tracking |
| **ExpanderAgent** | Content expansion | Expansion ratio tracking |
| **TranslatorAgent** | Multilingual translation | Cultural sensitivity, translator notes |
| **ReasoningAgent** | Chain-of-thought | Step validation with conclusion checks |
| **MergerAgent** | Multi-output synthesis | Conflict detection [CONFLICT: ...] |
| **RouterAgent** | Task classification | JSON category/complexity/agent routing |
| **MemoryAgent** | Context compression | Preserves key decisions and open questions |
| **ToolAgent** | Function call formatting | Structured JSON tool invocations |
| **JudgeAgent** | Quality arbiter | Multi-dimensional scoring (accuracy/completeness/clarity) |
| **RefinerAgent** | Iterative improvement | # REFINED: change tracking |
| **ClassifierAgent** | Domain/intent classification | Domain + urgency + agent recommendations |
| **ExtractorAgent** | Structured data extraction | Schema-based JSON extraction |
| **NarratorAgent** | Technical-to-readable prose | Analogies and examples for accessibility |
| **EnsembleAgent** | Multi-model voting | Agreement analysis, majority voting |

### 12 Self-Healing Audit Agents

| Agent | Scans For | Technique |
|---|---|---|
| **SyntaxAuditAgent** | Syntax errors | AST parsing (free) + LLM analysis |
| **LogicFlawAgent** | Logical contradictions | Off-by-one, infinite loops, bad conditionals |
| **HallucinationHunterAgent** | Fabricated facts | Per-claim confidence + cross-reference |
| **SecurityAuditAgent** | OWASP vulnerabilities | 8 static regex patterns + LLM deep scan |
| **PerformanceProfilerAgent** | Performance bottlenecks | O(n²), N+1, blocking I/O detection |
| **ConsistencyGuardAgent** | Cross-agent contradictions | Multi-output comparison |
| **TypeSafetyAgent** | Type mismatches | Missing annotations, None-dereference |
| **EdgeCaseHunterAgent** | Missing edge cases | Empty/null/overflow/race conditions |
| **TestGeneratorAgent** | Missing tests | Auto-generates pytest suites |
| **OutputVerifierAgent** | Task satisfaction | 0-100 score with retry logic |
| **SelfReflectionAgent** | Pipeline quality | Meta-cognitive audit (logged only) |
| **PatchApplierAgent** | Fix application | Surgical minimal patches with `# PATCHED:` |

### The Self-Healing Loop

```
User Task
    ↓
[Router] → classify task type
    ↓
[Planner] → decompose into subtasks
    ↓
[Workers] → execute subtasks (Coder, Researcher, etc.)
    ↓
[Audit Loop] ← up to 3 iterations
    │  ├─ SyntaxAudit → finds issues?
    │  ├─ SecurityAudit → finds vulnerabilities?
    │  ├─ TypeSafety → finds type errors?
    │  ├─ LogicFlaw → finds contradictions?
    │  ├─ OutputVerifier → score < 70?
    │  │
    │  └─ YES → PatchApplier → fix → re-audit
    │
    └─ PASS → return to user
```

---

## CLI Commands

```bash
# Serving
pravaha serve <model> [--quantize 4bit] [--tui] [--swarm] [--rag]

# Interactive Chat
pravaha chat [--server URL] [--model NAME]

# Benchmarks
pravaha bench [--model NAME] [--runs N]

# Model Management
pravaha models list
pravaha models info <model>
pravaha models pull <model>

# Swarm
pravaha swarm run "Write a REST API" --pipeline code-review
pravaha swarm list-agents
pravaha swarm pipeline plan-execute-audit

# RAG
pravaha rag ingest ./docs/
pravaha rag query "How does caching work?"
pravaha rag list

# Debug
pravaha debug replay <request-id>
pravaha debug step <request-id> --pos 42
pravaha debug trace <request-id>

# Plugins
pravaha plugin list
pravaha plugin install ./my-plugin/
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/v1/completions` | Text completion (OpenAI compatible) |
| POST | `/v1/chat/completions` | Chat completion (OpenAI compatible) |
| GET | `/v1/models` | List loaded models |
| POST | `/v1/vision/complete` | Multimodal vision + text |
| POST | `/v1/swarm/run` | Execute swarm pipeline |
| GET | `/v1/swarm/agents` | List all agents |
| GET | `/v1/swarm/pipelines` | List built-in pipelines |
| POST | `/v1/rag/ingest` | Ingest documents |
| GET | `/v1/rag/query` | Query vector store |
| POST | `/v1/branch` | Fork conversation |
| GET | `/v1/branch/{id}` | List branches |
| POST | `/v1/debug/replay` | Replay recorded request |
| GET | `/v1/debug/trace` | Export decision trace |
| POST | `/admin/reload` | Hot-reload config |
| POST | `/admin/lora/load` | Load LoRA adapter |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |
| WS | `/ws/generate` | WebSocket streaming |

---

## Built-in Pipelines

| Pipeline | Workers | Best For |
|---|---|---|
| `plan-execute-audit` | Planner → Coder → Critic | General coding tasks |
| `research-summarize` | Researcher → Reasoning → Summarizer | Research synthesis |
| `code-review` | Coder → Debugger → Critic → Refiner | Code quality |
| `creative-write` | Narrator → Expander → Refiner | Creative writing |
| `extract-classify` | Extractor → Classifier → Validator | Data extraction |

---

## Configuration

```yaml
# configs/default.yaml
model:
  model_path: gpt2
  device: auto
  quantization: null       # null | 4bit | 8bit
  max_seq_len: 2048

sampling:
  temperature: 0.7
  top_k: 50
  top_p: 0.9
  max_new_tokens: 256

swarm:
  enabled: true
  max_iterations: 3
  min_score: 70.0

rag:
  enabled: false
  embedding_model: all-MiniLM-L6-v2
```

---

## Project Structure

```
pravaha/
├── engine/            # AsyncPravahaEngine, events, background loop
├── decoder/           # Autoregressive decoder, sampling pipeline
├── scheduler/         # Continuous batching scheduler
├── memory/            # PagedKVCache, BlockManager, SessionCache
├── models/            # Model loader, architecture configs, weights, GGUF
├── tokenizer/         # HuggingFace tokenizer wrapper
├── quantization/      # INT4/INT8 quantization via bitsandbytes
├── swarm/
│   ├── agents/        # 32 individual agent files (20 workers + 12 auditors)
│   ├── orchestrator.py
│   ├── audit_loop.py
│   ├── pipeline.py
│   └── shared_memory.py
├── rag/               # Chunker, embedder, retriever, vector store
├── vision/            # Multimodal detector, preprocessor, engine
├── branching/         # Conversation branching manager
├── serving/
│   ├── app.py         # FastAPI factory (11 route modules + 3 middleware)
│   ├── middleware.py   # RequestID, Timing, ErrorHandler, RateLimit
│   ├── websocket.py   # WebSocket streaming
│   └── routes/        # completions, chat, models, swarm, rag, vision,
│                      #   branches, debug, admin, metrics, health
├── cli/
│   ├── main.py        # Typer entry point
│   ├── ascii_art.py   # Banners, gauges, grids
│   └── commands/      # serve, chat, bench, models, swarm, rag, debug, plugins
├── tui/
│   ├── app.py         # Textual dashboard
│   ├── pravaha.tcss   # Dark terminal theme
│   └── panels/        # 8 panels: header, chat, metrics, queue, swarm,
│                      #   audit, rag, log
├── plugins/           # Plugin base, loader, registry
├── guardrails/        # Content filter, token budget
├── debug/             # Request replayer, step debugger, trace logger
├── observability/     # Prometheus, tracer, cost estimator, self-benchmark
├── config/            # Engine config, YAML loading
└── cache/             # Multi-level caching utilities
```

---

## Proof of Work

### Benchmarks (GPT-2 baseline)

| Metric | FP16 | 4-bit NF4 | Improvement |
|---|---|---|---|
| VRAM Usage | 238 MB | **119 MB** | **50% reduction** |
| Inference Latency | ~0.08s/tok | ~0.09s/tok | Minimal penalty |
| 4 Concurrent Users | 4.4s total | **1.0s total** | **4.4x faster** |

### Demos

- **Phase 1-7**: [See demo videos in releases](https://github.com/pravaha/pravaha/releases)

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=pravaha

# Lint
ruff check pravaha/

# Type check
mypy pravaha/ --ignore-missing-imports
```

---

## License

MIT — Built with ❤️ by [@EternalcoderTanishq3](https://github.com/EternalcoderTanishq3)

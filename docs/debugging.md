# Pravāha v3.3 Debugging & Profiling Guide

This document provides a staff-engineer reference for introspecting, profiling, and debugging **Pravāha v3.3**. Pravāha's architecture spans multiple operational planes—including Rust block memory allocation, continuous batch scheduling, 52-agent ReAct swarm execution, sandboxed tool execution, and security audit loggers. This guide covers diagnostics for every layer of the system.

---

## 1. Diagnostic Architecture Overview

Debugging in Pravāha v3.3 is structured around four primary telemetry pillars:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Pillar 1: Request Replay & Deterministic Introspection                 │
│ Exact request state capture, seed initialization, trace re-execution   │
├─────────────────────────────────────────────────────────────────────────┤
│ Pillar 2: Token-Level & Logit Sampling Debugger                         │
│ Position-by-position logit extraction, top-K probabilities, penalty audit│
├─────────────────────────────────────────────────────────────────────────┤
│ Pillar 3: Memory & Continuous Scheduler Diagnostics                      │
│ Rust BlockAllocator stats, PrefixTrie hit ratio, KV-cache preemption     │
├─────────────────────────────────────────────────────────────────────────┤
│ Pillar 4: Swarm ReAct & Tool Sandbox Introspection                      │
│ Agent THINK->ACT->OBSERVE tracing, DockerSandbox logs, Audit Loop diffs│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Request Replay Engine (`pravaha debug replay`)

The Request Replay framework allows developers to capture failing requests in production and re-execute them locally under identical engine state conditions.

### Replay Mechanism

When a request is submitted, Pravāha records:
1. Exact prompt token IDs and decoding parameters.
2. Initial seed and temperature settings.
3. System prompt context and active agent configuration.

### CLI Usage

```bash
# Replay a specific request ID
pravaha debug replay req-9411-4f32

# Replay with verbose token-by-token output and custom temperature override
pravaha debug replay req-9411-4f32 --verbose --temperature 0.0
```

### API Endpoint (`POST /v1/debug/replay`)

```bash
curl -X POST http://localhost:8000/v1/debug/replay \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${PRAVAHA_API_KEY}" \
  -H "X-User-Role: operator" \
  -d '{
    "request_id": "req-9411-4f32",
    "override_params": {
      "temperature": 0.0,
      "max_tokens": 128
    }
  }'
```

### Python Programmatic Replay Script

```python
from pravaha.engine.async_engine import AsyncPravahaEngine
from pravaha.serving.routes.debug import ReplayEngine

async def run_replay_diagnostic(request_id: str, engine: AsyncPravahaEngine):
    replayer = ReplayEngine(engine=engine)
    
    print(f"[*] Loading historical request state for: {request_id}")
    request_state = await replayer.load_trace(request_id)
    
    print(f"[*] Original Prompt: {request_state['prompt']}")
    print(f"[*] Original Seed: {request_state['seed']}")
    
    # Execute replay pass
    result = await replayer.execute_replay(request_state)
    
    print(f"[+] Replay status: {result['status']}")
    print(f"[+] Matching tokens: {result['token_match_percentage']}%")
    if not result['matches_original']:
        print(f"[-] Divergence detected at position: {result['first_divergence_pos']}")
```

---

## 3. Token-Level Introspection (`pravaha debug step`)

Token-level step debugging allows developers to freeze generation at any step $N$ and inspect raw logits, applied penalties, and sampling probabilities.

### CLI Step Debug Command

```bash
pravaha debug step req-9411-4f32 --pos 42
```

### Sample Step Diagnostic Output

```
┌─ Token Debugger: Position 42 ───────────────────────────────────────────┐
│ Target Request ID : req-9411-4f32                                       │
│ Active Prompt Length: 128 tokens                                        │
│ Current Generated : 41 tokens                                           │
├─────────────────────────────────────────────────────────────────────────┤
│ Top 10 Candidate Logits (Post Temperature & Penalty Scaling):           │
│                                                                         │
│  Rank │ Token ID │ Token Text   │ Raw Logit │ Prob (%) │ Action        │
│ ──────┼──────────┼──────────────┼───────────┼──────────┼────────────── │
│    1  │     2041 │  " memory"   │   12.845  │   68.42% │ Selected      │
│    2  │     8192 │  " storage"  │   10.211  │   14.81% │ Filtered (Top-P)│
│    3  │      412 │  " cache"    │    9.804  │    9.85% │ Filtered      │
│    4  │      319 │  " buffer"   │    7.112  │    3.10% │ Filtered      │
│    5  │     1904 │  " pool"     │    5.981  │    1.24% │ Filtered      │
│    6  │    11204 │  " block"    │    5.110  │    0.62% │ Filtered      │
│    7  │       12 │  " ."        │    4.890  │    0.50% │ Filtered      │
│    8  │     9811 │  " region"   │    4.112  │    0.23% │ Filtered      │
│    9  │     5012 │  " space"    │    3.890  │    0.14% │ Filtered      │
│   10  │       99 │  " layout"   │    3.110  │    0.09% │ Filtered      │
├─────────────────────────────────────────────────────────────────────────┤
│ Active Logit Modifiers:                                                 │
│   - Temperature        : 0.70                                           │
│   - Top-P (Nucleus)    : 0.90                                           │
│   - Repetition Penalty : 1.15 (Applied to 14 active tokens)              │
│   - Frequency Penalty  : 0.00                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Full Trace Export & Diagnostic JSON Schema

Export complete trace metadata for offline analysis or ingestion into log aggregators:

```bash
pravaha debug trace req-9411-4f32 --out trace_req9411.json
```

### Structure of `trace_req9411.json`

```json
{
  "request_id": "req-9411-4f32",
  "timestamp": "2026-07-24T14:46:00.124Z",
  "metadata": {
    "model": "meta-llama/Llama-3-8B-Instruct",
    "user_role": "operator",
    "client_ip": "10.0.4.12"
  },
  "performance": {
    "ttft_ms": 42.1,
    "total_latency_ms": 310.5,
    "generation_tokens": 84,
    "tokens_per_second": 270.5
  },
  "scheduler_telemetry": {
    "queue_delay_ms": 1.2,
    "kv_blocks_allocated": 12,
    "prefix_cache_hit": true,
    "shared_prefix_length": 64
  },
  "generation_trace": [
    {
      "step": 0,
      "selected_token_id": 2041,
      "selected_token_text": " memory",
      "probability": 0.6842,
      "entropy": 0.812
    }
  ]
}
```

---

## 5. Memory & KV-Cache Introspection (Rust Core)

When Pravāha experiences high concurrency, KV-cache fragmentation or block preemption can cause latency degradation. You can profile the low-level Rust `BlockAllocator` and `PrefixTrie` engines directly.

### Python Introspection Snippet

```python
from pravaha_rust import BlockAllocator, PrefixTrie

# Inspect Rust Allocator Statistics
allocator = BlockAllocator(num_blocks=1024, block_size=16)

# Query runtime metrics
stats = allocator.get_stats()
print(f"[+] Total Blocks        : {stats.total_blocks}")
print(f"[+] Free Blocks         : {stats.free_blocks}")
print(f"[+] Allocated Blocks    : {stats.allocated_blocks}")
print(f"[+] Memory Utilization  : {stats.utilization():.2%}")
print(f"[+] Cache Hit Ratio     : {stats.hit_rate():.2%}")
print(f"[+] Alloc/Free Ratio    : {stats.alloc_free_ratio():.4f}")
```

### Diagnosing KV-Cache Preemption Stack Trace

Under severe VRAM memory pressure, the continuous scheduler preempts low-priority sequences and swaps their blocks to host RAM.

```
[2026-07-24 14:46:12.890] [ERROR] [pravaha.memory.block_manager] KV-Cache VRAM Exhaustion!
Traceback (most recent call last):
  File "/app/pravaha/memory/block_manager.py", line 142, in allocate_blocks
    block_ids = self._rust_allocator.batch_allocate(required_blocks)
RuntimeError: AllocationFailed: Insufficient free blocks (requested 16, available 2).

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/app/pravaha/engine/scheduler.py", line 210, in _step_scheduling
    self._preempt_lowest_priority_sequence()
  File "/app/pravaha/engine/scheduler.py", line 285, in _preempt_lowest_priority_sequence
    swapped_blocks = self.block_manager.swap_out(victim_seq.id)
  File "/app/pravaha/memory/block_manager.py", line 198, in swap_out
    logger.warning(
        f"[PREEMPTION] Evicted seq_id={victim_seq.id} ({len(swapped_blocks)} blocks) to CPU swap."
    )
```

**Resolution Procedure**:
1. Increase `kv_cache.swap_space_gb` in configuration (e.g., from `16.0` to `32.0`).
2. Reduce `scheduler.max_num_seqs` to bound sequence concurrency.
3. Enable 4-bit/8-bit KV-cache quantization if supported.

---

## 6. Swarm ReAct Agent & Tool Sandbox Debugging

Pravāha's 52-agent swarm relies on explicit ReAct loops. Debugging agent execution requires tracing internal thoughts, tool inputs, sandbox responses, and self-healing audit passes.

### Inspecting ReAct Trace

```python
from pravaha.swarm.orchestrator import SwarmOrchestrator
from pravaha.swarm.agents.worker_agents import PythonDeveloperAgent

# Run single agent with verbose debug logging enabled
agent = PythonDeveloperAgent()
agent.verbose = True

task = "Write a function to validate email addresses using regex."
result = await agent.run_react(task=task, context={}, engine=engine)

# Access step-by-step ReAct transcript
for step_idx, step in enumerate(result.react_steps):
    print(f"\n--- Step {step_idx + 1} ---")
    print(f"Thought    : {step.thought}")
    print(f"Action     : {step.tool_name}({step.tool_args})")
    print(f"Observation: {step.observation}")
```

### Debugging Docker Sandbox Execution Failure

When an agent executes code using `execute_python` tool, code runs inside `DockerSandbox`. Below is a diagnostic stack trace from a sandboxed execution timeout:

```
[2026-07-24 14:47:05.412] [WARNING] [pravaha.swarm.tools.docker_sandbox] Execution timed out!
Traceback (most recent call last):
  File "/app/pravaha/swarm/tools/docker_sandbox.py", line 88, in run_code
    container.wait(timeout=self.timeout_seconds)
  File "/opt/venv/lib/python3.11/site-packages/docker/models/containers.py", line 512, in wait
    raise urllib3.exceptions.ReadTimeoutError(self.client, url, "Read timed out.")
docker.errors.ContainerError: Sandbox execution exceeded hard limit of 5.0 seconds.

During handling of the above exception, the agent observation was generated:
Observation: Error: Code execution timed out after 5.0 seconds. Process terminated.
```

**Self-Healing Loop Recovery**:
Upon receiving the timeout observation, the self-healing auditor loop catches the failure, triggers `PatchApplier`, and prompts the agent to optimize the algorithm (e.g., eliminating infinite loops).

---

## 7. Memory Leak & PyTorch VRAM Profiling

When operating LLM engines under continuous load, distinguishing between PyTorch memory pool caching and actual memory leaks is critical.

### Python Tracemalloc & VRAM Profiler Script

```python
import torch
import tracemalloc
import logging
from pravaha.engine.async_engine import AsyncPravahaEngine

logger = logging.getLogger(__name__)

def profile_vram_and_host_memory(engine: AsyncPravahaEngine):
    tracemalloc.start()
    
    snapshot_before = tracemalloc.take_snapshot()
    vram_allocated_before = torch.cuda.memory_allocated() / (1024 ** 2)
    vram_reserved_before = torch.cuda.memory_reserved() / (1024 ** 2)
    
    logger.info(f"VRAM Allocated Before: {vram_allocated_before:.2f} MB")
    logger.info(f"VRAM Reserved Before : {vram_reserved_before:.2f} MB")
    
    # Run test workload...
    
    snapshot_after = tracemalloc.take_snapshot()
    top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    
    logger.info("[*] Top Host RAM Memory Allocations:")
    for stat in top_stats[:5]:
        logger.info(stat)
        
    vram_allocated_after = torch.cuda.memory_allocated() / (1024 ** 2)
    vram_reserved_after = torch.cuda.memory_reserved() / (1024 ** 2)
    
    logger.info(f"VRAM Allocated Delta : {vram_allocated_after - vram_allocated_before:.2f} MB")
    logger.info(f"VRAM Reserved Delta  : {vram_reserved_after - vram_reserved_before:.2f} MB")
```

---

## 8. WebSocket Streaming & Real-Time Connection Debugging

WebSocket stream interruptions can result from network proxies timing out idle TCP connections or sequence buffer overflow.

### WebSocket Connection Diagnostic Log Trace

```
[2026-07-24 14:47:18.910] [WARNING] [pravaha.serving.websocket] Client disconnected abruptly during token streaming.
Traceback (most recent call last):
  File "/app/pravaha/serving/websocket.py", line 64, in stream_tokens
    await websocket.send_text(json.dumps(token_payload))
  File "/opt/venv/lib/python3.11/site-packages/starlette/websockets.py", line 84, in send_text
    raise RuntimeError("Cannot call 'send' once a close message has been sent or received.")
RuntimeError: Cannot call 'send' once a close message has been sent or received.

[2026-07-24 14:47:18.915] [INFO] [pravaha.engine.async_engine] Cancelling sequence seq-7781-bc01 due to client disconnect.
```

---

## 9. Structured JSON Logging & Context Tracking

Pravāha standardizes logging via `StructuredLogger` (`pravaha/observability/structured_logger.py`). Logs automatically carry `X-Request-ID` across async task boundaries.

### Log Output Example with Exception Trace

```json
{
  "timestamp": "2026-07-24T14:47:30.512Z",
  "level": "ERROR",
  "logger": "pravaha.serving.middleware",
  "request_id": "req-8f92a10e-9411",
  "user_role": "user",
  "client_ip": "192.168.1.105",
  "message": "Unhandled error caught in ErrorHandlerMiddleware",
  "exception": {
    "type": "PermissionDeniedError",
    "message": "RBAC Denied: User with role 'user' attempted to access endpoint '/v1/swarm/execute' requiring 'operator' role.",
    "traceback": [
      "File \"/app/pravaha/serving/middleware.py\", line 52, in dispatch\n    return await call_next(request)",
      "File \"/app/pravaha/serving/rbac.py\", line 69, in _role_checker\n    raise HTTPException(status_code=403, detail=...)"
    ]
  }
}
```

---

## 10. Circuit Breaker & Failure Recovery Profiling

When internal engine errors accumulate, `CircuitBreaker` trips to `OPEN` state to prevent cascading service failure.

### Diagnostic Exception Stack Trace

```
[2026-07-24 14:48:02.119] [CRITICAL] [pravaha.engine.circuit_breaker] Circuit Breaker TRIPPED to OPEN!
Traceback (most recent call last):
  File "/app/pravaha/engine/async_engine.py", line 302, in generate
    self.circuit_breaker.verify_state()
  File "/app/pravaha/engine/circuit_breaker.py", line 75, in verify_state
    raise CircuitBreakerOpenException(
        f"Circuit breaker is OPEN. Consecutive failures ({self.failure_count}) "
        f"exceeded threshold ({self.config.failure_threshold}). "
        f"Recovery window remaining: {self.remaining_cooldown_seconds:.1f}s"
    )
pravaha.engine.circuit_breaker.CircuitBreakerOpenException: Circuit breaker is OPEN. Recovery window remaining: 28.4s
```

### Manual Circuit Breaker Reset Command

```bash
# Force reset circuit breaker state via CLI
pravaha debug reset-circuit-breaker --server http://localhost:8000
```

---

## 11. Prometheus Metrics Reference

Pravāha exposes runtime metrics at `/metrics`.

| Metric Name | Type | Description |
| :--- | :--- | :--- |
| `pravaha_requests_total` | Counter | Total incoming HTTP/WebSocket requests partitioned by status code and endpoint. |
| `pravaha_tokens_generated_total` | Counter | Cumulative tokens generated by `AsyncPravahaEngine`. |
| `pravaha_ttft_seconds` | Histogram | Time to First Token (TTFT) distribution in seconds. |
| `pravaha_inter_token_latency_seconds` | Histogram | Inter-token generation latency (TBT) distribution. |
| `pravaha_kv_cache_usage_ratio` | Gauge | Current ratio of allocated physical KV blocks (0.0 to 1.0). |
| `pravaha_active_sequences` | Gauge | Current count of active sequences in the continuous scheduler. |
| `pravaha_swarm_agent_runs_total` | Counter | Total agent ReAct runs partitioned by agent role and completion status. |
| `pravaha_circuit_breaker_state` | Gauge | State of circuit breaker (`0` = CLOSED, `1` = HALF_OPEN, `2` = OPEN). |

---

## 12. Diagnostic Troubleshooting Quick Reference

```
Problem: High TTFT (Time to First Token)
├── Check 1: Inspect Rust prefix trie cache hit rate via `/metrics`.
│   └── If hit_rate < 0.20, check if system prompts are standardized across requests.
├── Check 2: Check KV-cache block allocation queue delay.
│   └── If queue delay > 10ms, lower `max_num_seqs` or increase GPU memory limit.
└── Check 3: Inspect model initialization precision.
    └── Ensure model is loaded in `float16` or `4bit`, not full `float32`.

Problem: Swarm ReAct Loop Stuck in Infinite Loop
├── Check 1: Inspect `max_react_steps` in agent configuration.
│   └── Bound max iterations (default: 8).
└── Check 2: Check `ToolRegistry` observation response.
    └── Verify tools return clear error messages rather than empty strings on failure.
```

# Pravāha v3.3 REST & Streaming API Reference

The Pravāha v3.3 API provides high-throughput, OpenAI-compatible text completion, chat, vision, multi-agent swarm orchestration, conversation branching, RAG retrieval, and real-time WebSocket token streaming. The API layer is implemented using FastAPI and Uvicorn, integrated directly with the `AsyncPravahaEngine` and the underlying continuous batching scheduler.

---

## 1. Architectural Overview & Request Lifecycle

```
[ Client Request ]
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│  FastAPI Application Layer (pravaha/serving/app.py)                     │
│  ├─ RequestIDMiddleware       --> Injects X-Request-ID (UUIDv4)        │
│  ├─ TimingMiddleware          --> Measures latency (X-Process-Time)    │
│  ├─ ErrorHandlerMiddleware    --> Encapsulates unhandled exceptions     │
│  ├─ RateLimitMiddleware       --> Sliding-window IP rate limiter       │
│  └─ BearerAuthMiddleware      --> Verifies PRAVAHA_API_KEY Bearer Token│
└────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Role-Based Access Control (pravaha/serving/rbac.py)                   │
│  └─ Depends(require_role(Role)) [ADMIN (3) > OPERATOR (2) > USER (1)]   │
└────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│  AsyncPravahaEngine & Scheduler (pravaha/engine/)                       │
│  ├─ RequestQueue & PagedKVCache Block Allocation                       │
│  ├─ Continuous Batching Loop (Daemon Thread)                          │
│  └─ CircuitBreaker Protection & Structured JSON Logger                 │
└────────────────────────────────────────────────────────────────────────┘
       │
       ▼
[ Token Streaming / SSE / JSON Response ]
```

### Thread Boundary & Async Architecture
The API server operates an asynchronous event loop handling HTTP and WebSocket I/O. Engine inference occurs via `AsyncPravahaEngine`, which bridges asyncio tasks to a dedicated background scheduler thread. Tokens cross the thread boundary safely via `loop.call_soon_threadsafe(queue.put_nowait, token)`.

---

## 2. Authentication & Authorization

### Bearer Token Authentication
Authentication is enforced globally via `BearerAuthMiddleware`. Requests must supply an HTTP `Authorization` header containing a valid Bearer token matching the server's `PRAVAHA_API_KEY` environment variable.

```http
Authorization: Bearer pr_live_9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c
```

#### Excluded Public Paths
The following endpoints bypass authentication for health checking and OpenAPI documentation rendering:
- `/health`
- `/health/ready`
- `/docs`
- `/openapi.json`
- `/redoc`

#### Authentication Failure Response (`401 Unauthorized`)
```json
{
  "error": {
    "message": "Missing or invalid API key",
    "type": "authentication_error",
    "code": "unauthorized",
    "param": null
  }
}
```

### Role-Based Access Control (RBAC)
Pravāha enforces role-based permissions across control plane endpoints and agent tool invocations. The user role is passed via the `X-User-Role` header or derived from administrative API keys.

#### Role Hierarchy Table

| Role | Weight Level | Endpoint Access Scope | Permitted Tool Execution |
| :--- | :---: | :--- | :--- |
| **`USER`** | 1 | Standard Completions, Chat, Vision, RAG Query, Branching | Standard read-only tools, web search |
| **`OPERATOR`** | 2 | Swarm Execution, Document Ingestion, Model Switching, Debug Replay | Code sandbox execution, file writing |
| **`ADMIN`** | 3 | Configuration Reload (`/admin/reload`), LoRA Load/Activate, Model Merging | Full system tools, shell execution, system reloads |

#### RBAC Authorization Denial (`403 Forbidden`)
```json
{
  "detail": "Permission denied: Requires 'admin' role."
}
```

---

## 3. Global HTTP Headers & Middleware

Every response returned by Pravāha v3.3 contains diagnostic metadata headers injected by the middleware pipeline:

| Header Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `X-Request-ID` | String (UUIDv4) | Unique request trace ID attached to structured log entries. | `req_8f1a2b3c-4d5e-6f7a-8b9c-0d1e2f3a4b5c` |
| `X-Process-Time` | String (ms) | Server processing time from request reception to response emission. | `42.8ms` |
| `X-Pravaha-Version` | String | Framework version string. | `3.3.0` |

---

## 4. Text Completions API

### `POST /v1/completions`

Generates text completions for a given prompt using continuous batching and PagedAttention KV-cache allocation.

#### Request Headers
```http
Content-Type: application/json
Authorization: Bearer <PRAVAHA_API_KEY>
X-User-Role: user
```

#### Request JSON Schema
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["model", "prompt"],
  "properties": {
    "model": {
      "type": "string",
      "description": "Model identifier registered in engine."
    },
    "prompt": {
      "type": "string",
      "description": "Text prompt to generate completions for."
    },
    "max_tokens": {
      "type": "integer",
      "default": 128,
      "minimum": 1,
      "maximum": 8192
    },
    "temperature": {
      "type": "number",
      "default": 0.7,
      "minimum": 0.0,
      "maximum": 2.0
    },
    "top_p": {
      "type": "number",
      "default": 0.9,
      "minimum": 0.0,
      "maximum": 1.0
    },
    "top_k": {
      "type": "integer",
      "default": 50,
      "minimum": 0
    },
    "stream": {
      "type": "boolean",
      "default": false
    },
    "session_id": {
      "type": ["string", "null"],
      "default": null,
      "description": "Optional session ID for prefix cache reuse."
    }
  }
}
```

#### Non-Streaming Response (`200 OK`)
```json
{
  "id": "cmpl-a9f8b7c6d5e4f3a2",
  "object": "text_completion",
  "created": 1774343000,
  "model": "meta-llama/Llama-3-8B",
  "choices": [
    {
      "text": "\n\nContinuous batching enables efficient throughput by scheduling requests at the token iteration boundary rather than waiting for sequence completion.",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 24,
    "total_tokens": 42
  }
}
```

---

## 5. Chat Completions API

### `POST /v1/chat/completions`

OpenAI-compatible chat completion endpoint supporting system, user, and assistant message structures.

#### Request Body Example
```json
{
  "model": "meta-llama/Llama-3-8B",
  "messages": [
    {
      "role": "system",
      "content": "You are a Staff Systems Engineer specializing in distributed LLM serving."
    },
    {
      "role": "user",
      "content": "Explain PagedAttention block allocation in Pravāha."
    }
  ],
  "max_tokens": 256,
  "temperature": 0.3,
  "stream": false
}
```

#### Server-Sent Events (SSE) Streaming (`stream: true`)
When `stream: true` is set, the server returns a `text/event-stream` response emitting data chunks:

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Request-ID: req_9a8b7c6d-5e4f

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1774343005,"model":"meta-llama/Llama-3-8B","choices":[{"index":0,"delta":{"role":"assistant","content":"PagedAttention"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1774343005,"model":"meta-llama/Llama-3-8B","choices":[{"index":0,"delta":{"content":" partitions"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1774343006,"model":"meta-llama/Llama-3-8B","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

## 6. Vision API

### `POST /v1/vision/complete`

Processes multimodal payloads combining high-resolution visual input with structured prompts.

#### Request JSON Schema
```json
{
  "model": "llava-1.5-7b",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Analyze this system architecture diagram and identify single points of failure."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}
```

#### Response JSON Schema
```json
{
  "id": "vis-3f2a1b0c9d8e",
  "object": "vision.completion",
  "created": 1774343010,
  "model": "llava-1.5-7b",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "The architecture diagram displays a single load balancer instance without active-passive failover, representing a primary single point of failure."
      },
      "finish_reason": "stop",
      "index": 0
    }
  ]
}
```

---

## 7. Swarm Intelligence API

The Swarm API orchestrates Pravāha’s 52 ReAct-based agents organized across workers, auditors, security specialists, and UI/UX designers.

### `POST /v1/swarm/run`
Executes an autonomous agent pipeline with an integrated self-healing audit loop.

#### Request Body
```json
{
  "prompt": "Refactor the authentication module to use HMAC SHA-256 tokens and implement rate limiting.",
  "pipeline": "security-hardened-code",
  "max_audit_iterations": 3,
  "min_score": 75.0,
  "context": {
    "repository": "pravaha/serving",
    "strict_mode": true
  }
}
```

#### Response Body
```json
{
  "execution_id": "swm_9876543210",
  "pipeline": "security-hardened-code",
  "status": "completed",
  "audit_passed": true,
  "iterations_used": 2,
  "final_score": 88.5,
  "output": "```python\n# Refactored authentication with HMAC SHA-256...\n```",
  "agent_steps": [
    {
      "step": 1,
      "agent": "SecurityAuditAgent",
      "action": "execute_python",
      "thought": "Analyzing token verification logic for timing attack vulnerabilities.",
      "observation": "Pass: Constant-time comparison implemented via hmac.compare_digest."
    }
  ],
  "audit_report": {
    "issues_found": 1,
    "issues_resolved": 1,
    "security_score": 92.0
  }
}
```

### `GET /v1/swarm/agents`
Lists all 52 registered swarm agents, their assigned roles, priorities, and attached tool capability sets.

### `GET /v1/swarm/pipelines`
Retrieves pre-defined agent workflows (`code-review`, `security-audit`, `design-system`, `full-stack-refactor`).

---

## 8. RAG & Vector Knowledge API

### `POST /v1/rag/ingest`
Ingests plain text, Markdown, or PDF documents into the FAISS vector database.

#### Request Body
```json
{
  "document_id": "doc_arch_spec_v3",
  "content": "Pravāha uses a Rust-powered PrefixTrie for O(k) prefix matching across KV-cache blocks...",
  "metadata": {
    "source": "ARCHITECTURE.md",
    "author": "Engineering Team"
  },
  "chunk_size": 512,
  "chunk_overlap": 64
}
```

#### Response (`200 OK`)
```json
{
  "status": "success",
  "document_id": "doc_arch_spec_v3",
  "chunks_created": 4,
  "total_tokens_indexed": 1840
}
```

### `GET /v1/rag/query`
Performs dense vector retrieval against the vector index.

#### Query Parameters
- `query` (string, required): Search query.
- `top_k` (integer, default 5): Number of matches to return.
- `similarity_threshold` (float, default 0.70): Cosine similarity filter score.

#### Response Example
```json
{
  "query": "How does prefix sharing work?",
  "results": [
    {
      "chunk_id": "doc_arch_spec_v3_chunk_0",
      "score": 0.892,
      "text": "Pravāha uses a Rust-powered PrefixTrie for O(k) prefix matching across KV-cache blocks...",
      "metadata": { "source": "ARCHITECTURE.md" }
    }
  ]
}
```

---

## 9. Conversation Branching API

Allows developers to fork chat histories into Directed Acyclic Graph (DAG) branches, maximizing prefix reuse via PagedAttention.

### `POST /v1/branch`
Forks a session history at a designated message index.

```json
{
  "session_id": "sess_main_001",
  "fork_at": 3,
  "label": "optimization-experiment"
}
```

#### Response (`201 Created`)
```json
{
  "branch_id": "br_opt_exp_992",
  "session_id": "sess_main_001",
  "fork_index": 3,
  "parent_branch_id": "main",
  "created_at": 1774343020
}
```

### `GET /v1/branch/{session_id}`
Lists all active branch nodes associated with a session ID.

### `POST /v1/branch/{branch_id}/checkout`
Activates a specific branch node for subsequent completion calls.

---

## 10. Debug, Replay & Tracing API

### `POST /v1/debug/replay`
Replays a historic inference request deterministically using recorded engine state and random seeds.

### `GET /v1/debug/step`
Steps through inference execution token by token, returning KV-cache memory block maps and sampling logit distributions.

---

## 11. Control Plane & Admin API

Administrative operations require the `ADMIN` role (`X-User-Role: admin` or admin key).

### `POST /admin/reload`
Hot-reloads engine configurations (sampling rules, scheduler queue limits, guardrails) without stopping the server.

```json
{
  "sampling": {
    "temperature": 0.4
  },
  "scheduler": {
    "max_batch_size": 64
  }
}
```

### `POST /admin/lora/load` & `POST /admin/lora/activate`
Dynamically loads and activates Low-Rank Adaptation (LoRA) weight matrices into GPU memory.

---

## 12. Real-Time WebSocket API

### Endpoint: `WS /ws/generate`

Provides full-duplex, low-latency token streaming over standard WebSockets.

#### Protocol Flow

1. **Client Connection:** `ws://localhost:8000/ws/generate`
2. **Client Sends Request Frame (JSON):**
```json
{
  "prompt": "Write a Rust function for fast string hashing.",
  "max_tokens": 150,
  "temperature": 0.2
}
```
3. **Server Streams Token Frames (JSON):**
```json
{"token": "pub ", "done": false, "step": 1}
{"token": "fn ", "done": false, "step": 2}
{"token": "hash_str", "done": false, "step": 3}
```
4. **Server Terminal Frame:**
```json
{"token": "", "done": true, "finish_reason": "stop", "total_tokens": 48}
```

---

## 13. System Health & Observability API

### `GET /health`
Returns readiness and liveness status, including internal engine stats and memory stats.

```json
{
  "status": "healthy",
  "engine": "AsyncPravahaEngine",
  "version": "3.3.0",
  "uptime_seconds": 86400,
  "kv_cache": {
    "gpu_blocks_used": 128,
    "gpu_blocks_free": 896,
    "block_utilization_pct": 12.5
  },
  "circuit_breakers": {
    "model_inference": "closed",
    "docker_sandbox": "closed"
  }
}
```

### `GET /metrics`
Exports Prometheus-formatted performance metrics for scraping:
- `pravaha_requests_total{status="200"}`
- `pravaha_time_to_first_token_seconds_bucket`
- `pravaha_kv_cache_block_utilization_ratio`
- `pravaha_circuit_breaker_tripped_total`

---

## 14. Error Handling & Circuit Breaker Fault Isolation

Errors emitted by the API strictly follow a standardized JSON envelope structure:

```json
{
  "error": {
    "message": "CircuitBreaker 'docker_sandbox' is OPEN. Call rejected.",
    "type": "CircuitBreakerOpenError",
    "code": "service_unavailable",
    "param": null
  }
}
```

### Standard HTTP Status Codes

| Code | Status Name | Cause / Trigger |
| :---: | :--- | :--- |
| `200` | OK | Request processed successfully. |
| `401` | Unauthorized | Missing or invalid `Authorization: Bearer` header. |
| `403` | Forbidden | Insufficient RBAC privileges (`X-User-Role` mismatch). |
| `429` | Too Many Requests | Sliding-window IP rate limit exceeded (`RateLimitMiddleware`). |
| `500` | Internal Server Error | Unhandled engine exception caught by `ErrorHandlerMiddleware`. |
| `503` | Service Unavailable | Circuit breaker tripped open due to repeated upstream failures. |

---

## 15. Complete Client Integration Examples

### Python (`httpx` + `asyncio`)

```python
import asyncio
import httpx

API_URL = "http://localhost:8000/v1/chat/completions"
API_KEY = "pr_live_9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c"

async def generate_chat():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "X-User-Role": "user",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/Llama-3-8B",
        "messages": [{"role": "user", "content": "Explain continuous batching."}],
        "max_tokens": 100
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(API_URL, headers=headers, json=payload, timeout=30.0)
        print(f"Status: {response.status_code}")
        print("Response:", response.json())

if __name__ == "__main__":
    asyncio.run(generate_chat())
```

### JavaScript (`fetch` with Server-Sent Events)

```javascript
async function streamChat() {
  const response = await fetch("http://localhost:8000/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": "Bearer pr_live_9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c",
      "X-User-Role": "user"
    },
    body: JSON.stringify({
      model: "meta-llama/Llama-3-8B",
      messages: [{ role: "user", content: "Write a quicksort in JavaScript." }],
      stream: true
    })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value);
    console.log("Chunk:", chunk);
  }
}
```

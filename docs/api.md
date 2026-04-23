# API Reference

All endpoints are OpenAI-compatible unless noted otherwise.

## Completions

### POST `/v1/completions`

Text completion endpoint.

```json
{
  "model": "gpt2",
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "stream": false,
  "session_id": null
}
```

**Response:**

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1234567890,
  "model": "gpt2",
  "choices": [{"text": "...", "finish_reason": "stop", "index": 0}],
  "usage": {"prompt_tokens": 4, "completion_tokens": 100, "total_tokens": 104}
}
```

## Chat Completions

### POST `/v1/chat/completions`

Chat completion with message history.

```json
{
  "model": "gpt2",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true
}
```

## Models

### GET `/v1/models`

List available models.

## Vision

### POST `/v1/vision/complete`

Multimodal vision + text completion.

```json
{
  "model": "llava-1.5-7b",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What's in this image?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }]
}
```

## Swarm

### POST `/v1/swarm/run`

Execute a swarm pipeline.

```json
{
  "prompt": "Write a REST API for user management",
  "pipeline": "code-review",
  "max_audit_iterations": 3,
  "min_score": 70.0
}
```

### GET `/v1/swarm/agents`

List all loaded agents with stats.

### GET `/v1/swarm/pipelines`

List built-in pipeline definitions.

## RAG

### POST `/v1/rag/ingest`

Ingest a document into the vector store.

### GET `/v1/rag/query?query=...&top_k=5`

Query the vector store.

### GET `/v1/rag/sources`

List ingested documents.

## Branching

### POST `/v1/branch`

Fork a conversation at a specific message index.

### GET `/v1/branch/{session_id}`

List all branches for a session.

### POST `/v1/branch/{branch_id}/checkout`

Switch to a branch.

### DELETE `/v1/branch/{branch_id}`

Delete a branch.

## Debug

### POST `/v1/debug/replay`

Replay a recorded request exactly.

### GET `/v1/debug/step?request_id=...&pos=0`

Step through inference at a token position.

### GET `/v1/debug/trace?request_id=...`

Export full decision trace.

## Admin

### POST `/admin/reload`

Hot-reload configuration without restart.

### POST `/admin/lora/load`

Load a LoRA adapter.

### POST `/admin/lora/activate`

Activate a loaded adapter.

### POST `/admin/merge`

Queue a model merge (SLERP).

## System

### GET `/health`

Health check. Returns engine status and stats.

### GET `/metrics`

Prometheus-format metrics.

### WebSocket `/ws/generate`

Real-time token streaming.

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/generate");
ws.send(JSON.stringify({prompt: "Hello", max_tokens: 50}));
ws.onmessage = (e) => {
  const data = JSON.parse(e.data);
  if (!data.done) process.stdout.write(data.token);
};
```

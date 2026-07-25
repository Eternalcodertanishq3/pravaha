## Configuration Reference


Pravāha is configured via YAML configuration files located in `configs/`:

### Master Engine Configuration (`configs/engine_default.yaml`)

```yaml
# Pravāha Master Configuration Specification v3.3

engine:
  model_name: "gpt2"
  device: "cuda"
  dtype: "float16"
  max_model_len: 2048
  gpu_memory_utilization: 0.85
  block_size: 16

scheduler:
  max_num_seqs: 256
  max_waiting_tokens: 4096
  max_waiting_queue_len: 1000
  max_swapped_queue_len: 500
  max_finished_queue_len: 1000
  overload_threshold: 0.95

serving:
  host: "0.0.0.0"
  port: 8000
  api_key_env_var: "PRAVAHA_API_KEY"
  rate_limit_per_min: 100
  cors_origins: ["*"]

security:
  enable_auth: true
  enable_rbac: true
  enable_sandbox: true
  sandbox_type: "docker"
  docker_memory_mb: 512
  docker_cpus: 1.0

observability:
  log_level: "INFO"
  json_logging: true
  redact_secrets: true
  redact_pii: true
  enable_audit_trail: true
```

### Swarm Orchestration Configuration (`configs/swarm_default.yaml`)

```yaml
# Swarm Agent Topology Specification
swarm_defaults:
  max_agent_steps: 10
  tool_timeout_s: 15
  max_tool_retries: 2
  enable_context_locking: true

agents:
  - name: "researcher"
    role: "Information Retrieval Agent"
    allowed_tools: ["web_fetcher"]
    max_steps: 8
  - name: "coder"
    role: "Python Software Engineer Agent"
    allowed_tools: ["python_repl", "bash_tool"]
    max_steps: 10

pipelines:
  - id: "code_gen_dag"
    nodes: ["researcher", "coder"]
    edges:
      - from: "researcher"
        to: "coder"
```

---



## REST API Specifications


In addition to standard OpenAI-compatible endpoints, Pravāha provides enterprise administration and monitoring APIs:

### 1. Chat Completions Endpoint
`POST /v1/chat/completions`

```json
{
  "model": "gpt2",
  "messages": [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Write a Python function for binary search."}
  ],
  "max_tokens": 100,
  "temperature": 0.2,
  "stream": true
}
```

### 2. Health & Readiness Endpoint
`GET /health/ready`

```json
{
  "status": "ready",
  "version": "3.3.0",
  "timestamp": "2026-07-24T14:40:00Z",
  "subsystems": {
    "engine": "healthy",
    "scheduler": "healthy",
    "kv_cache": "healthy",
    "circuit_breaker": "CLOSED"
  },
  "metrics": {
    "waiting_queue_depth": 0,
    "allocated_blocks": 12,
    "gpu_memory_allocated_mb": 398.5
  }
}
```

### 3. Admin User Data Export (GDPR)
`POST /admin/export_user_data`

```json
{
  "user_id": "usr_99823"
}
```

### 4. Admin User Data Deletion (GDPR Right-to-be-Forgotten)
`POST /admin/delete_user`

```json
{
  "user_id": "usr_99823",
  "confirm_permanent_delete": true
}
```

---



## CLI & Operational Tooling


Pravāha includes a rich suite of command-line tools for development, operational management, benchmarking, and emergency maintenance:

### 1. Server CLI (`serve.py`)
```bash
# Run server with HTTPS and custom GPU reservation
python serve.py \
  --host 0.0.0.0 \
  --port 8443 \
  --gpu-memory-utilization 0.90 \
  --ssl-keyfile certs/key.pem \
  --ssl-certfile certs/cert.pem
```

### 2. Emergency Rollback CLI (`scripts/rollback.py`)
```bash
# Rollback to main branch and verify readiness probe
python scripts/rollback.py --target main --verify
```

### 3. Production Benchmark CLI (`scripts/run_production_soak_test.py`)
```bash
# Execute concurrency benchmark suite (1, 5, 10, 25, 50 streams)
python scripts/run_production_soak_test.py
```

### 4. Empirical Security & Fault Drill CLI (`scripts/generate_evidence_dossier.py`)
```bash
# Execute queue saturation, circuit breaker, and security probe drills
python scripts/generate_evidence_dossier.py --run-all
```

---



## Production Deployment Manifests


### 1. Docker Compose Manifest (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  pravaha-engine:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PRAVAHA_API_KEY=your_secure_api_key_here
      - PRAVAHA_CORS_ORIGINS=https://app.yourdomain.com
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/ready"]
      interval: 10s
      timeout: 5s
      retries: 3
```

### 2. Enterprise Kubernetes Manifest (`k8s/deployment.yaml`)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pravaha-engine
  namespace: pravaha-system
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pravaha-engine
  template:
    metadata:
      labels:
        app: pravaha-engine
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
    spec:
      containers:
      - name: pravaha
        image: pravaha/engine:v3.3.0
        ports:
        - containerPort: 8000
        resources:
          limits:
            cpu: "8"
            memory: "16Gi"
            nvidia.com/gpu: "1"
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 15
```

---



## Python SDK & Client Integration Patterns


Pravāha is 100% API-compatible with the OpenAI Python SDK and supports native `asyncio` streaming clients as well as LangChain / LlamaIndex custom provider integrations:

### 1. Standard OpenAI Python Client Integration

```python
import openai

# Configure client to point to local or production Pravāha endpoint
client = openai.OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="YOUR_PRAVAHA_API_KEY",  # Or set PRAVAHA_API_KEY env var
)

# Execute streaming chat completion
response = client.chat.completions.create(
    model="gpt2",
    messages=[
        {"role": "system", "content": "You are a helpful software architecture assistant."},
        {"role": "user", "content": "Compare continuous batching vs naive static batching."}
    ],
    max_tokens=150,
    temperature=0.7,
    stream=True,
)

print("Pravāha Stream Output: ", end="")
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")
```

---

### 2. Native Async `httpx` SSE Streaming Client

```python
import asyncio
import json
import httpx

async def stream_from_pravaha():
    url = "http://127.0.0.1:8000/v1/chat/completions"
    headers = {
        "Authorization": "Bearer YOUR_PRAVAHA_API_KEY",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt2",
        "messages": [{"role": "user", "content": "Write a Python decorator for rate limiting."}],
        "max_tokens": 100,
        "temperature": 0.2,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                print(f"Error {response.status_code}: {await response.aread()}")
                return

            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    content = data["choices"][0]["delta"].get("content", "")
                    print(content, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(stream_from_pravaha())
```

---

### 3. Cryptographic Audit Ledger Tamper Verification Script

To audit and cryptographically verify the integrity of Pravāha's append-only SHA-256 audit ledger:

```python
from pravaha.observability.audit_trail import AuditTrail

# Initialize Audit Trail with log path
audit = AuditTrail(log_path="logs/audit_ledger.log")

# Verify SHA-256 hash chain integrity
is_valid, corrupted_index = audit.verify_integrity()

if is_valid:
    print("✅ AUDIT LEDGER INTEGRITY VERIFIED: SHA-256 hash chain is 100% intact.")
else:
    print(f"❌ TAMPER DETECTED! Audit record at index {corrupted_index} has been altered.")
```

---



## Prometheus Alert Rules Specification (`docker/rules.yml`)


Pravāha includes pre-configured Prometheus alert definitions for enterprise monitoring and PagerDuty integration:

```yaml
groups:
  - name: pravaha_alerts
    rules:
      - alert: PravahaHighQueueLatency
        expr: pravaha_ttft_seconds{quantile="0.95"} > 0.500
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High P95 Time-To-First-Token Latency"
          description: "P95 TTFT has exceeded 500ms for more than 2 minutes."

      - alert: PravahaQueueSaturation
        expr: pravaha_waiting_queue_depth > 800
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Scheduler Waiting Queue Saturation"
          description: "Waiting queue depth has exceeded 80% capacity (800/1000)."

      - alert: PravahaHighErrorRate
        expr: rate(pravaha_requests_total{status=~"5.."}[5m]) / rate(pravaha_requests_total[5m]) > 0.05
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "Elevated 5xx Server Error Rate"
          description: "Server error rate exceeded 5% over a 5-minute window."

      - alert: PravahaCircuitBreakerOpen
        expr: pravaha_circuit_breaker_state == 1
        for: 30s
        labels:
          severity: warning
        annotations:
          summary: "Dependency Circuit Breaker Open"
          description: "Circuit breaker entered OPEN state due to upstream dependency failures."
```

---




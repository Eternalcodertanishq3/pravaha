# Pravāha Command-Line Interface (CLI) Reference Manual

This document provides a complete reference for Pravāha's command-line interface tools, entry points, administrative utilities, benchmark scripts, and troubleshooting commands.

---

## 1. Primary Entry Point (`serve.py`)

The primary entry point for launching the Pravāha LLM serving engine and API server.

### Command Syntax

```bash
python serve.py [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--host` | `str` | `127.0.0.1` | Network interface address to bind the HTTP server. Use `0.0.0.0` for containerized or network exposure. |
| `--port` | `int` | `8000` | Port number on which the server listens for HTTP/HTTPS requests. |
| `--model` | `str` | `gpt2` | Name or path of the HuggingFace transformer model to load. |
| `--device` | `str` | `cuda` | Target compute device (`cuda`, `cpu`, or `mps`). |
| `--dtype` | `str` | `float16` | Model weights precision (`float16`, `bfloat16`, `float32`). |
| `--gpu-memory-utilization` | `float` | `0.85` | Fraction of total GPU VRAM reserved for PagedAttention KV-cache blocks. |
| `--max-num-seqs` | `int` | `256` | Maximum number of concurrent sequences handled in the scheduler. |
| `--block-size` | `int` | `16` | Number of tokens per PagedAttention physical KV block. |
| `--ssl-keyfile` | `str` | `None` | Path to TLS private key file for HTTPS. |
| `--ssl-certfile` | `str` | `None` | Path to TLS certificate file for HTTPS. |
| `--tui` | `flag` | `False` | Launch interactive Terminal User Interface (TUI) instead of plain server logs. |
| `--config` | `str` | `None` | Path to custom YAML configuration file (overrides CLI flags). |
| `--log-level` | `str` | `info` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |

### Execution Examples

#### Example 1.1: Basic Development Server
```bash
python serve.py --host 127.0.0.1 --port 8000 --model gpt2
```

#### Example 1.2: Production HTTPS Server with Custom VRAM Limit
```bash
python serve.py \
  --host 0.0.0.0 \
  --port 8443 \
  --model gpt2 \
  --gpu-memory-utilization 0.90 \
  --ssl-keyfile /etc/pravaha/certs/key.pem \
  --ssl-certfile /etc/pravaha/certs/cert.pem \
  --log-level info
```

#### Example 1.3: Terminal User Interface (TUI) Mode
```bash
python serve.py --tui --model gpt2
```

---

## 2. Emergency Rollback Utility (`scripts/rollback.py`)

The rollback tool allows operators to revert the Pravāha codebase to a known stable git tag or commit hash while automatically verifying system readiness via health probes.

### Command Syntax

```bash
python scripts/rollback.py [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--target` | `str` | `main` | Git commit hash, tag, or branch to checkout. |
| `--verify` | `flag` | `True` | Execute post-checkout readiness check (`GET /health/ready`). |
| `--health-url` | `str` | `http://127.0.0.1:8000/health/ready` | Readiness endpoint URL for verification. |
| `--timeout` | `int` | `30` | Maximum seconds to wait for health probe success after rollback. |
| `--force` | `flag` | `False` | Discard uncommitted local changes before checkout. |

### Execution Examples

#### Example 2.1: Rollback to Main Branch with Readiness Verification
```bash
python scripts/rollback.py --target main --verify
```

#### Example 2.2: Emergency Force Rollback to Tag `v3.2.0`
```bash
python scripts/rollback.py --target v3.2.0 --force --timeout 60
```

---

## 3. Production Benchmark & Soak Test Runner (`scripts/run_production_soak_test.py`)

Runs multi-tenant streaming inference benchmarks across concurrency levels (1, 5, 10, 25, 50) and collects high-resolution hardware telemetry.

### Command Syntax

```bash
python scripts/run_production_soak_test.py [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--concurrencies` | `str` | `1,5,10,25,50` | Comma-separated list of concurrent stream tiers to benchmark. |
| `--tokens-per-request` | `int` | `20` | Number of new tokens to generate per streaming request. |
| `--iterations` | `int` | `10` | Number of repeated trial iterations per concurrency tier ($n=10$). |
| `--output-json` | `str` | `None` | File path to write raw JSON benchmark results. |

### Execution Examples

#### Example 3.1: Standard Benchmark Run
```bash
python scripts/run_production_soak_test.py
```

#### Example 3.2: Export Detailed Telemetry JSON
```bash
python scripts/run_production_soak_test.py \
  --concurrencies 1,10,50 \
  --tokens-per-request 50 \
  --iterations 20 \
  --output-json benchmark_results.json
```

---

## 4. Empirical Evidence & Security Drill Generator (`scripts/generate_evidence_dossier.py`)

Executes synthetic queue saturation drills, circuit breaker fault injections, SHA-256 audit ledger checks, and adversarial security probes.

### Command Syntax

```bash
python scripts/generate_evidence_dossier.py [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--run-all` | `flag` | `True` | Execute all 4 verification suites (Queue, Circuit Breaker, Audit, Security). |
| `--queue-capacity` | `int` | `500` | Target queue capacity for saturation drill. |
| `--queue-requests` | `int` | `700` | Total concurrent request submissions for queue saturation drill. |
| `--audit-records` | `int` | `500` | Number of hash-chained audit log records to generate and verify. |

### Execution Examples

```bash
python scripts/generate_evidence_dossier.py --run-all
```

---

## 5. Environment & System Diagnostic Utility (`scripts/diagnostics.py`)

Prints physical GPU capabilities, PyTorch CUDA build flags, memory allocations, and Rust toolchain integration status.

### Command Syntax

```bash
python -c "import pravaha; pravaha.print_diagnostics()"
```

### Sample Output

```text
================================================================================
                    PRAVĀHA SYSTEM DIAGNOSTIC REPORT
================================================================================
OS Platform              : Windows-11-10.0.26200-SP0 (x86_64)
Python Executable        : C:\Personal Projects\Pravāha\.venv\Scripts\python.exe
Python Version           : 3.11.0
PyTorch Version          : 2.6.0+cu124
CUDA Available           : True
CUDA Version             : 12.4
GPU Device Name          : NVIDIA GeForce RTX 4050 Laptop GPU
Total VRAM (GB)          : 6.00 GB
CPU Physical Cores       : 14 Physical / 20 Logical
Total System RAM (GB)    : 15.73 GB
Rust Extension Status    : Active (rustc 1.93.0)
PagedAttention Allocator : Rust Native PrefixTrie Accelerator
================================================================================
```

---

## 6. CLI Workflows & Shell Automation

### Shell Script for Automated CI/CD Readiness Checks

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] Checking Python environment..."
python --version

echo "[2/4] Executing automated test suite..."
python -m pytest tests/ -v

echo "[3/4] Running empirical security and queue drills..."
python scripts/generate_evidence_dossier.py

echo "[4/4] Starting server dry-run..."
python serve.py --host 127.0.0.1 --port 8009 &
SERVER_PID=$!

sleep 5

echo "Probing readiness endpoint..."
curl -f http://127.0.0.1:8009/health/ready

kill -9 $SERVER_PID
echo "CI/CD Verification Passed Successfully!"
```

---

## 7. Troubleshooting CLI Errors

### Error: `Address already in use`
- **Cause:** Port 8000 is occupied by another process.
- **Fix:** Specify a different port using `--port 8005` or terminate the occupying process:
  ```bash
  # Windows
  netstat -ano | findstr :8000
  taskkill /PID <PID> /F

  # Linux
  lsof -i :8000
  kill -9 <PID>
  ```

### Error: `CUDA out of memory`
- **Cause:** VRAM utilization threshold is set too high for available GPU memory.
- **Fix:** Lower VRAM reservation:
  ```bash
  python serve.py --gpu-memory-utilization 0.70 --max-num-seqs 128
  ```

---

## 8. Environment Variables Reference Table

Pravāha supports configuration override via environment variables. Environment variables take precedence over default YAML values but are overridden by explicit CLI flags.

| Environment Variable | Type | Default | Description |
|---|---|---|---|
| `PRAVAHA_HOST` | `str` | `127.0.0.1` | Network interface address to bind HTTP server. |
| `PRAVAHA_PORT` | `int` | `8000` | Port number on which HTTP server listens. |
| `PRAVAHA_MODEL_NAME` | `str` | `gpt2` | Target HuggingFace model identifier. |
| `PRAVAHA_DEVICE` | `str` | `cuda` | Primary execution device (`cuda` or `cpu`). |
| `PRAVAHA_DTYPE` | `str` | `float16` | Model floating point precision (`float16`, `bfloat16`, `float32`). |
| `PRAVAHA_GPU_MEMORY_UTILIZATION` | `float` | `0.85` | Fraction of total VRAM allocated for PagedAttention. |
| `PRAVAHA_MAX_NUM_SEQS` | `int` | `256` | Maximum active sequences in scheduler. |
| `PRAVAHA_BLOCK_SIZE` | `int` | `16` | Number of tokens per physical KV block. |
| `PRAVAHA_API_KEY` | `str` | `None` | Authentication secret key for `BearerAuthMiddleware`. |
| `PRAVAHA_CORS_ORIGINS` | `str` | `*` | Comma-separated list of allowed CORS origins. |
| `PRAVAHA_RATE_LIMIT_PER_MIN` | `int` | `100` | Rate limit threshold per IP address. |
| `PRAVAHA_LOG_LEVEL` | `str` | `INFO` | Logging output level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `PRAVAHA_SSL_KEYFILE` | `str` | `None` | TLS private key file path. |
| `PRAVAHA_SSL_CERTFILE` | `str` | `None` | TLS certificate file path. |

---

## 9. Docker CLI & Container Management Commands

### 9.1 Build Docker Container Image

```bash
docker build -t pravaha/engine:v3.3.0 -f docker/Dockerfile .
```

### 9.2 Run Container with GPU Access

```bash
docker run -d \
  --name pravaha-container \
  --gpus all \
  -p 8000:8000 \
  -e PRAVAHA_API_KEY="prod_secret_key_99823" \
  -e PRAVAHA_CORS_ORIGINS="https://ai.company.com" \
  pravaha/engine:v3.3.0
```

### 9.3 Container Health Inspection

```bash
# View container status and health probe outputs
docker inspect --format='{{json .State.Health}}' pravaha-container | jq .
```

---

## 10. Curl REST API Command Cheat-Sheet

### 10.1 Health Check Endpoint
```bash
curl -i http://127.0.0.1:8000/health
```

### 10.2 Deep Readiness Check Endpoint
```bash
curl -i http://127.0.0.1:8000/health/ready
```

### 10.3 Non-Streaming Chat Completion
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "gpt2",
    "messages": [
      {"role": "user", "content": "What is continuous batching?"}
    ],
    "max_tokens": 50,
    "temperature": 0.0,
    "stream": false
  }'
```

### 10.4 Streaming Chat Completion (Server-Sent Events)
```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "gpt2",
    "messages": [
      {"role": "user", "content": "Write a short poem about AI infrastructure."}
    ],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": true
  }'
```

### 10.5 Admin Export User Data Endpoint (GDPR)
```bash
curl -X POST http://127.0.0.1:8000/admin/export_user_data \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -d '{
    "user_id": "usr_77123"
  }'
```

### 10.6 Admin Delete User Data Endpoint (GDPR Right-to-be-Forgotten)
```bash
curl -X POST http://127.0.0.1:8000/admin/delete_user \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -d '{
    "user_id": "usr_77123",
    "confirm_permanent_delete": true
  }'
```

---

## 11. Complete Python Programmatic Engine Usage

In addition to command-line execution, Pravāha can be imported directly into Python applications:

```python
import asyncio
from pravaha.config.engine_config import EngineConfig
from pravaha.engine.async_engine import AsyncPravahaEngine
from pravaha.decoder.sampling import SamplingParams

async def main():
    # 1. Instantiate configuration
    config = EngineConfig(
        model_name="gpt2",
        device="cuda",
        dtype="float16",
        gpu_memory_utilization=0.85,
        block_size=16,
        max_num_seqs=256,
    )

    # 2. Build engine via Factory
    engine = AsyncPravahaEngine(config=config)

    # 3. Define sampling params
    sampling_params = SamplingParams(
        max_new_tokens=40,
        temperature=0.7,
        top_p=0.9,
    )

    # 4. Stream generated tokens
    prompt = "Continuous scheduling optimizes GPU utilization by"
    print(f"Prompt: {prompt}\nResponse: ", end="")

    async for token in engine.generate(prompt, sampling_params):
        print(token, end="", flush=True)

    print("\n\nEngine Execution Finished Successfully.")

    # 5. Clean shutdown
    engine.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Summary

The Pravāha CLI and environment interfaces provide comprehensive, operational controls for local development, automated testing, container orchestration, and production serving environments.


# Configuration Guide

Pravāha uses YAML configuration files with Pydantic validation.

## Config Files

```
configs/
├── default.yaml          # Full default configuration
├── phase1.yaml           # Minimal CPU testing
├── swarm_default.yaml    # Swarm agent configuration
└── rag_default.yaml      # RAG pipeline configuration
```

## Usage

```bash
pravaha serve gpt2 --config configs/default.yaml
```

## Model Configuration

```yaml
model:
  model_path: gpt2                    # HuggingFace model ID or local path
  device: auto                        # auto | cpu | cuda | cuda:0
  quantization: null                  # null | 4bit | 8bit
  max_seq_len: 2048                   # Maximum sequence length
  trust_remote_code: false            # Allow remote code execution
  torch_dtype: float16                # float16 | bfloat16 | float32
```

## Sampling Configuration

```yaml
sampling:
  temperature: 0.7                    # 0.0 = deterministic, 1.0+ = creative
  top_k: 50                           # Keep top K tokens (0 = disabled)
  top_p: 0.9                          # Nucleus sampling threshold
  max_new_tokens: 256                 # Maximum output tokens
  repetition_penalty: 1.1             # Penalize repeated tokens
```

## Cache Configuration

```yaml
cache:
  block_size: 16                      # Tokens per cache block
  num_gpu_blocks: 256                 # GPU blocks (0 = auto)
  max_sessions: 1000                  # Multi-turn session cache
  session_ttl_seconds: 3600           # Session expiry (1 hour)
```

## Scheduler Configuration

```yaml
scheduler:
  max_batch_size: 32                  # Maximum concurrent batch size
  max_waiting: 256                    # Maximum queued requests
```

## Swarm Configuration

```yaml
swarm:
  enabled: true                       # Enable/disable swarm
  max_iterations: 3                   # Audit loop iterations
  min_score: 70.0                     # Minimum satisfaction score
  default_pipeline: plan-execute-audit
  agent_roles: [planner, coder, ...]  # Enabled agent roles
  audit_roles: [syntax_audit, ...]    # Enabled auditor roles
```

## RAG Configuration

```yaml
rag:
  enabled: false
  embedding_model: all-MiniLM-L6-v2
  chunk_size: 512
  chunk_overlap: 64
  top_k: 5
  similarity_threshold: 0.7
  vector_store:
    type: faiss
    save_path: ./data/rag/index.faiss
```

## Hot Reload

Some settings can be changed without restart:

```bash
curl -X POST http://localhost:8000/admin/reload \
  -d '{"sampling": {"temperature": 0.5}}'
```

Hot-reloadable: `sampling.*`, `scheduler.*`, `swarm.*`, `guardrails.*`

Not hot-reloadable: `model.*`, `cache.*` (require restart)

# Pravāha v3.3 Complete Configuration Reference

Pravāha v3.3 employs a unified, type-safe configuration system driven by **YAML configuration files** and validated strictly via **Pydantic schemas** (`pravaha/config.py`). This framework supports dynamic environment variable substitution (`${ENV_VAR}`), hot-reloading of non-structural parameters, and distinct deployment profiles ranging from edge CPU deployments to multi-GPU high-throughput clusters.

---

## 1. Configuration Architecture & Precedence

Configurations are parsed at initialization and injected throughout the engine stack. The configuration loader merges default settings with user-supplied files, environment variable overrides, and explicit CLI flags.

```
                           Configuration Loading Sequence
                                         │
                                         ▼
                      [ Default Config (configs/default.yaml) ]
                                         │
                                         ▼
                      [ User YAML File (via --config / -c) ]
                                         │
                                         ▼
                      [ Environment Variable Overrides ]
                                         │
                                         ▼
                      [ CLI Flags (e.g. --port, --quantize) ]
                                         │
                                         ▼
                      [ Pydantic Schema Validation Pass ]
                                         │
                                         ▼
                      [ Instantiated Engine Config Object ]
```

### Environment Variable Interpolation
Values inside YAML configuration files can reference environment variables using `${VARIABLE_NAME}` or `${VARIABLE_NAME:-default_value}` syntax:

```yaml
security:
  bearer_token: "${PRAVAHA_API_KEY:-default_secret_key}"
  docker_sandbox:
    memory_limit: "${DOCKER_MEM_LIMIT:-256MB}"
```

### Type-Safe Pydantic Model Hierarchy
```python
# System Configuration Model Architecture (pravaha/config.py)
class ModelConfig(BaseModel):
    model_path: str = Field(default="gpt2")
    device: str = Field(default="auto")
    quantization: Optional[str] = Field(default=None)
    max_seq_len: int = Field(default=2048, ge=128, le=32768)
    trust_remote_code: bool = Field(default=False)
    torch_dtype: str = Field(default="float16")
    cpu_threads: int = Field(default=8, ge=1)

class SamplingConfig(BaseModel):
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_new_tokens: int = Field(default=256, ge=1)
    repetition_penalty: float = Field(default=1.1, ge=0.0)

class PravahaConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    swarm: SwarmConfig = Field(default_factory=SwarmConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
```

---

## 2. Config Presets Overview

Pravāha includes several curated configuration presets under `configs/`:

| Preset File | Primary Target Environment | Key Feature Scenarios |
| :--- | :--- | :--- |
| `configs/default.yaml` | Standard Production (GPU / High-Memory CPU) | Continuous batching, PagedAttention, 52-agent swarm, structured logging enabled. |
| `configs/phase1.yaml` | Minimal Dev / CPU Testing | Small KV-cache footprint, swarm disabled, benchmark disabled, fast boot. |
| `configs/swarm_default.yaml` | Multi-Agent Swarm Workflows | All 52 agents enabled, ReAct step budget set to 10, self-healing audit loop active. |
| `configs/rag_default.yaml` | Document Retrieval Workflows | FAISS vector store enabled, sentence-transformers embedding, chunk size 512. |

---

## 3. Top-Level Schema Categories

The configuration object is composed of 9 primary section models:

```
PravahaConfig
 ├── model: ModelConfig
 ├── sampling: SamplingConfig
 ├── cache: CacheConfig
 ├── scheduler: SchedulerConfig
 ├── swarm: SwarmConfig
 ├── rag: RAGConfig
 ├── security: SecurityConfig
 ├── observability: ObservabilityConfig
 └── circuit_breaker: CircuitBreakerConfig
```

---

## 4. Section-by-Section Schema Reference

### 4.1 Model Configuration (`model`)

Configures model weights, hardware device allocation, and quantization parameters.

```yaml
model:
  model_path: "meta-llama/Llama-3-8B"  # HuggingFace model ID or local directory
  device: "auto"                        # Execution device: auto | cpu | cuda | cuda:0
  quantization: "4bit"                  # Weight quantization: null | 4bit | 8bit
  max_seq_len: 4096                     # Maximum context sequence length (tokens)
  trust_remote_code: false              # Allow execution of custom code from HuggingFace
  torch_dtype: "float16"                # PyTorch tensor precision: float16 | bfloat16 | float32
  cpu_threads: 8                        # Number of CPU threads for PyTorch operations
```

| Field | Type | Default | Validation / Constraints |
| :--- | :---: | :--- | :--- |
| `model_path` | String | `"gpt2"` | Non-empty string. |
| `device` | String | `"auto"` | Must be one of: `auto`, `cpu`, `cuda`, `cuda:N`. |
| `quantization` | String / Null | `null` | Must be one of: `null`, `none`, `4bit`, `8bit`. |
| `max_seq_len` | Integer | `2048` | Minimum: `128`, Maximum: `32768`. |
| `trust_remote_code` | Boolean | `false` | Security warning logged if set to `true`. |
| `torch_dtype` | String | `"float16"` | Must be one of: `float16`, `bfloat16`, `float32`. |
| `cpu_threads` | Integer | `8` | Minimum: `1`, Maximum: CPU core count. |

---

### 4.2 Sampling Configuration (`sampling`)

Controls default token generation parameters for completions.

```yaml
sampling:
  temperature: 0.7            # Randomness control: 0.0 (greedy) to 2.0
  top_k: 50                   # Nucleus top-k token selection (0 to disable)
  top_p: 0.9                  # Top-p cumulative probability threshold
  max_new_tokens: 256         # Maximum tokens generated per request
  repetition_penalty: 1.1     # Penalty factor for repeating tokens (> 1.0)
  presence_penalty: 0.0       # Penalty for token presence in generated text
  frequency_penalty: 0.0      # Penalty based on token frequency in generated text
  seed: null                  # Random seed for deterministic generation (null = random)
```

| Field | Type | Default | Validation / Constraints |
| :--- | :---: | :--- | :--- |
| `temperature` | Float | `0.7` | Range: `0.0` to `2.0`. |
| `top_k` | Integer | `50` | Minimum: `0`. `0` disables top-k sampling. |
| `top_p` | Float | `0.9` | Range: `0.0` to `1.0`. |
| `max_new_tokens` | Integer | `256` | Minimum: `1`. Upper bounded by `max_seq_len`. |
| `repetition_penalty` | Float | `1.1` | Range: `1.0` (disabled) to `2.0`. |

---

### 4.3 Memory & KV-Cache Configuration (`cache`)

Configures PagedAttention block allocation, prefix caching, and Rust `PrefixTrie` acceleration.

```yaml
cache:
  block_size: 16              # Tokens per KV-cache block (must be power of 2)
  num_gpu_blocks: 1024        # Number of GPU cache blocks (0 = auto-calculate)
  num_cpu_blocks: 512         # Number of swapped CPU cache blocks
  max_sessions: 1000          # Maximum tracked multi-turn session histories
  session_ttl_seconds: 3600   # Idle session expiry duration (seconds)
  prefix_sharing: true        # Enable zero-copy prefix sharing across requests
  rust_trie_enabled: true     # Use Rust-powered PrefixTrie for O(k) prefix matching
```

| Field | Type | Default | Validation / Constraints |
| :--- | :---: | :--- | :--- |
| `block_size` | Integer | `16` | Must be a power of 2 (e.g., `8`, `16`, `32`, `64`). |
| `num_gpu_blocks` | Integer | `1024` | `0` triggers automatic GPU VRAM allocation math. |
| `num_cpu_blocks` | Integer | `512` | Memory host block allocation count. |
| `prefix_sharing` | Boolean | `true` | Enables zero-copy shared prefix KV blocks. |
| `rust_trie_enabled` | Boolean | `true` | Uses Rust `PrefixTrie` instead of Python hash lookup. |

---

### 4.4 Scheduler Configuration (`scheduler`)

Controls continuous batching queue behavior, batch sizes, and preemption policies.

```yaml
scheduler:
  max_batch_size: 32          # Maximum requests batched into a single GPU iteration
  max_waiting: 256            # Maximum queued requests before rejecting with HTTP 429
  preemption_mode: "recompute" # Preemption strategy: recompute | swap
  waiting_timeout_seconds: 60.0 # Queue timeout before request aborts
  continuous_batching_interval_ms: 2.0 # Iteration scheduler loop sleep interval
```

---

### 4.5 Swarm Agent Configuration (`swarm`)

Controls the 52 ReAct-based swarm agents, self-healing audit loops, and container sandboxing.

```yaml
swarm:
  enabled: true               # Enable or disable the 52-agent swarm system
  max_iterations: 3           # Maximum self-healing audit loop iterations
  min_score: 70.0             # Minimum score required for output release
  default_pipeline: "full-stack-refactor" # Default pipeline workflow
  docker_sandbox_enabled: true # Enforce Docker container isolation for code execution
  tool_timeout_seconds: 10.0   # Maximum execution timeout for tool calls
  agent_roles:                # List of active agent roles
    - "planner"
    - "coder"
    - "auditor"
    - "security_scanner"
```

---

### 4.6 RAG & Retrieval Configuration (`rag`)

Configures vector store embeddings and chunking strategies.

```yaml
rag:
  enabled: false              # Enable or disable RAG functionality
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 512             # Document chunk token length
  chunk_overlap: 64           # Overlap between adjacent chunks
  top_k: 5                    # Number of document contexts retrieved per query
  similarity_threshold: 0.70  # Minimum cosine similarity score threshold
  vector_store:
    type: "faiss"             # Vector store engine: faiss | memory
    save_path: "./data/rag/index.faiss" # Persistence storage path
```

---

### 4.7 Security & Access Control (`security`)

Configures authentication, RBAC authorization, and execution sandboxing bounds.

```yaml
security:
  api_key_required: true      # Enforce Bearer token verification
  bearer_token: "${PRAVAHA_API_KEY}"
  rbac_enabled: true          # Enforce Role-Based Access Control
  docker_sandbox:
    cpu_limit: "1.0"          # Maximum CPU cores per container execution
    memory_limit: "256MB"     # Maximum RAM per container execution
    network_mode: "none"      # Network isolation: none | host | bridge
    read_only_rootfs: true    # Enforce read-only root filesystem in sandbox
  pii_redaction: true         # Automatically redact SSNs, emails, API keys in logs
```

---

### 4.8 Observability & Audit Trail (`observability`)

Configures logging, Prometheus metrics, OpenTelemetry tracing, and SHA-256 audit chaining.

```yaml
observability:
  log_level: "INFO"           # Logging verbosity: DEBUG | INFO | WARNING | ERROR
  log_format: "json"          # Output format: json | text
  structured_logging: true    # Contextual logging with X-Request-ID propagation
  audit_trail_path: "audit_trail.jsonl" # Path to SHA-256 hash-chained log
  sha256_verification: true   # Auto-verify audit log integrity on boot
  prometheus_port: 9090       # Prometheus metrics server port (0 = disable)
  tracer_enabled: true        # Enable OpenTelemetry request tracing
```

---

### 4.9 Circuit Breaker Configuration (`circuit_breaker`)

Configures fault isolation behavior to protect engine stability under component failures.

```yaml
circuit_breaker:
  failure_threshold: 5        # Failures before tripping from CLOSED to OPEN
  recovery_timeout_seconds: 30.0 # Time to wait before entering HALF_OPEN probe state
  half_open_success_threshold: 2 # Successes in HALF_OPEN required to close circuit
```

---

## 5. Fully Commented Production Configuration YAML

```yaml
# ====================================================================
# Pravāha v3.3 Enterprise Production Configuration
# ====================================================================

model:
  model_path: "meta-llama/Llama-3-8B"
  device: "cuda:0"
  quantization: "4bit"
  max_seq_len: 4096
  trust_remote_code: false
  torch_dtype: "float16"
  cpu_threads: 16

sampling:
  temperature: 0.5
  top_k: 40
  top_p: 0.95
  max_new_tokens: 512
  repetition_penalty: 1.05
  presence_penalty: 0.0
  frequency_penalty: 0.0
  seed: null

cache:
  block_size: 16
  num_gpu_blocks: 2048
  num_cpu_blocks: 1024
  max_sessions: 5000
  session_ttl_seconds: 7200
  prefix_sharing: true
  rust_trie_enabled: true

scheduler:
  max_batch_size: 64
  max_waiting: 512
  preemption_mode: "swap"
  waiting_timeout_seconds: 30.0
  continuous_batching_interval_ms: 1.0

swarm:
  enabled: true
  max_iterations: 3
  min_score: 75.0
  default_pipeline: "full-stack-refactor"
  docker_sandbox_enabled: true
  tool_timeout_seconds: 15.0
  agent_roles:
    - "planner"
    - "coder"
    - "auditor"
    - "security_audit"
    - "accessibility_auditor"

rag:
  enabled: true
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 512
  chunk_overlap: 64
  top_k: 5
  similarity_threshold: 0.75
  vector_store:
    type: "faiss"
    save_path: "./data/rag/production.faiss"

security:
  api_key_required: true
  bearer_token: "${PRAVAHA_API_KEY}"
  rbac_enabled: true
  docker_sandbox:
    cpu_limit: "2.0"
    memory_limit: "512MB"
    network_mode: "none"
    read_only_rootfs: true
  pii_redaction: true

observability:
  log_level: "INFO"
  log_format: "json"
  structured_logging: true
  audit_trail_path: "./logs/audit_trail.jsonl"
  sha256_verification: true
  prometheus_port: 9090
  tracer_enabled: true

circuit_breaker:
  failure_threshold: 5
  recovery_timeout_seconds: 45.0
  half_open_success_threshold: 3
```

---

## 6. Dynamic Configuration Hot-Reloading

Pravāha supports zero-downtime configuration updates for operational parameters via `POST /admin/reload`.

### Hot-Reload Classification Table

| Category | Parameters Permitted for Hot-Reload | Parameters Requiring Restart |
| :--- | :--- | :--- |
| **Sampling** | `temperature`, `top_k`, `top_p`, `repetition_penalty` | None |
| **Scheduler** | `max_batch_size`, `max_waiting`, `waiting_timeout_seconds` | `preemption_mode` |
| **Swarm** | `max_iterations`, `min_score`, `tool_timeout_seconds` | `enabled`, `docker_sandbox_enabled` |
| **Security** | `pii_redaction`, `rbac_enabled` | `bearer_token`, `docker_sandbox.*` |
| **Observability** | `log_level` | `prometheus_port`, `audit_trail_path` |
| **Model & Cache** | None (All require restart) | `model_path`, `device`, `quantization`, `block_size`, `num_gpu_blocks` |

#### Hot-Reload Command Example
```bash
curl -X POST http://localhost:8000/admin/reload \
  -H "Authorization: Bearer ${PRAVAHA_API_KEY}" \
  -H "X-User-Role: admin" \
  -H "Content-Type: application/json" \
  -d '{
    "sampling": { "temperature": 0.3 },
    "scheduler": { "max_batch_size": 128 }
  }'
```

---

## 7. Operational Hardware Profiles

To assist operators in tuning Pravāha for specific deployment environments, the following profiles detail optimal configuration presets.

### Profile A: High-Throughput NVIDIA A100 / H100 GPU Cluster
```yaml
model:
  device: "cuda:0"
  quantization: null
  torch_dtype: "bfloat16"
cache:
  block_size: 32
  num_gpu_blocks: 8192
  rust_trie_enabled: true
scheduler:
  max_batch_size: 128
  continuous_batching_interval_ms: 0.5
```

### Profile B: Edge CPU / Single Node Workstation
```yaml
model:
  device: "cpu"
  quantization: "4bit"
  cpu_threads: 16
cache:
  block_size: 16
  num_gpu_blocks: 0
  num_cpu_blocks: 1024
scheduler:
  max_batch_size: 8
  continuous_batching_interval_ms: 5.0
```

---

## 8. Configuration Validation & Error Handling

Pydantic enforces strict validation rules at startup. If invalid settings are detected, Pravāha outputs descriptive diagnostic errors and terminates gracefully:

### Common Pydantic Validation Errors

```
pydantic.ValidationError: 1 validation error for PravahaConfig
cache -> block_size
  value is not a valid power of 2 (got 18, expected one of [8, 16, 32, 64])
```

```
pydantic.ValidationError: 1 validation error for PravahaConfig
scheduler -> max_batch_size
  ensure this value is greater than 0 (got -5)
```

```
pydantic.ValidationError: 1 validation error for PravahaConfig
model -> torch_dtype
  unexpected value 'float64', expected one of ['float16', 'bfloat16', 'float32']
```

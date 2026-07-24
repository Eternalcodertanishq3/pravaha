# Pravāha v3.3 Enterprise Deployment Guide

This guide provides staff-engineer level instructions for deploying, scaling, and maintaining **Pravāha v3.3** in enterprise production environments. Pravāha v3.3 combines a continuous batching LLM inference engine featuring PagedAttention KV-caching with an autonomous 52-agent ReAct swarm, protected by enterprise security middleware (`BearerAuthMiddleware`, `RBACManager`, `DockerSandbox`, and `SHA256AuditTrail`).

---

## 1. Production Architecture Topology

In production environments, Pravāha operates as a scalable microservice layer behind high-performance ingress controllers. Below is the reference topology:

```
                               ┌─────────────────────────┐
                               │   Client Applications   │
                               └────────────┬────────────┘
                                            │ HTTPS / WSS
                                            ▼
                               ┌─────────────────────────┐
                               │ Enterprise API Gateway  │
                               │  (Nginx / Traefik / F5) │
                               └────────────┬────────────┘
                                            │ Bearer Auth / TLS
                                            ▼
                        ┌───────────────────────────────────────┐
                        │ Pravāha FastAPI Gateway Instances      │
                        │ (RequestID, Timing, BearerAuth, RBAC) │
                        └───────────────────┬───────────────────┘
                                            │
           ┌────────────────────────────────┼────────────────────────────────┐
           │                                │                                │
           ▼                                ▼                                ▼
┌───────────────────────┐        ┌───────────────────────┐        ┌───────────────────────┐
│ AsyncPravahaEngine    │        │ AsyncPravahaEngine    │        │ AsyncPravahaEngine    │
│ (Instance 1)          │        │ (Instance 2)          │        │ (Instance N)          │
├───────────────────────┤        ├───────────────────────┤        ├───────────────────────┤
│ Continuous Scheduler  │        │ Continuous Scheduler  │        │ Continuous Scheduler  │
│ PagedKVCache          │        │ PagedKVCache          │        │ PagedKVCache          │
│ Rust BlockAllocator   │        │ Rust BlockAllocator   │        │ Rust BlockAllocator   │
└──────────┬────────────┘        └──────────┬────────────┘        └──────────┬────────────┘
           │                                │                                │
           ├────────────────────────────────┼────────────────────────────────┘
           │                                │
           ▼                                ▼
┌───────────────────────┐        ┌───────────────────────┐
│  NVIDIA GPU Cluster   │        │ Docker Sandbox Worker │
│  (A100 / H100 SXM)    │        │ (Isolated Code Tools) │
└───────────────────────┘        └───────────────────────┘
           │                                │
           ▼                                ▼
┌───────────────────────┐        ┌───────────────────────┐
│  Prometheus Metrics   │        │ SHA-256 Audit Trail   │
│  & Grafana Dashboards │        │ Append-Only Storage   │
└───────────────────────┘        └───────────────────────┘
```

---

## 2. Environment Variables Matrix

All configuration parameters in Pravāha v3.3 can be injected via environment variables.

| Environment Variable | Default Value | Type | Description |
| :--- | :--- | :--- | :--- |
| `PRAVAHA_MODEL_PATH` | `meta-llama/Llama-3-8B-Instruct` | String | Model repository identifier or local absolute path. |
| `PRAVAHA_QUANTIZATION` | `4bit` | String | Quantization precision (`none`, `8bit`, `4bit`, `awq`). |
| `PRAVAHA_DEVICE` | `cuda` | String | Compute device execution target (`cuda`, `cpu`, `auto`). |
| `PRAVAHA_PORT` | `8000` | Integer | TCP port for FastAPI API server. |
| `PRAVAHA_WORKERS` | `4` | Integer | Uvicorn worker process count for API gateway. |
| `PRAVAHA_MAX_NUM_SEQS` | `256` | Integer | Maximum active sequence concurrency limit in continuous scheduler. |
| `PRAVAHA_MAX_BATCHED_TOKENS` | `8192` | Integer | Upper bound of batched tokens processed per iteration step. |
| `PRAVAHA_GPU_MEM_UTIL` | `0.85` | Float | Target VRAM fraction reserved for PagedAttention KV-cache blocks. |
| `PRAVAHA_BLOCK_SIZE` | `16` | Integer | Number of tokens per physical memory block allocated in Rust FFI. |
| `PRAVAHA_API_KEY` | `""` | String | Master API bearer token. If empty, authentication is disabled. |
| `PRAVAHA_ENABLE_RBAC` | `true` | Boolean | Enforce Role-Based Access Control (`ADMIN`, `OPERATOR`, `USER`). |
| `PRAVAHA_SWARM_ENABLED` | `true` | Boolean | Enable 52-agent autonomous ReAct swarm orchestration. |
| `PRAVAHA_SELF_HEAL` | `true` | Boolean | Enable multi-iteration audit and auto-patching loop. |
| `DOCKER_SANDBOX_IMAGE` | `pravaha/sandbox:latest` | String | Docker image used for isolating untrusted Python code tools. |
| `SHA256_AUDIT_LOG_DIR` | `./data/audit` | String | Directory path for cryptographically signed audit log ledger. |
| `CIRCUIT_BREAKER_THRESHOLD`| `5` | Integer | Failure count threshold before opening engine circuit breaker. |

---

## 3. Bare Metal & Systemd Production Deployment

For bare metal servers or GPU instances (e.g., AWS EC2 `g5.4xlarge` or `p4d.24xlarge`), running Pravāha under `systemd` ensures automatic process recovery, logging to `journald`, and resource boundary enforcement.

### 1. User & Directory Provisioning

```bash
# Create dedicated system user
sudo useradd -r -s /bin/false pravaha

# Create runtime and configuration directories
sudo mkdir -p /etc/pravaha /var/log/pravaha /var/lib/pravaha/data /var/lib/pravaha/audit
sudo chown -R pravaha:pravaha /var/log/pravaha /var/lib/pravaha
```

### 2. Systemd Service Definition (`/etc/systemd/system/pravaha.service`)

```ini
[Unit]
Description=Pravāha v3.3 Self-Healing LLM Engine & Swarm
After=network.target nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=simple
User=pravaha
Group=pravaha
WorkingDirectory=/var/lib/pravaha
Environment="PATH=/opt/pravaha/venv/bin:/usr/local/cuda/bin:/usr/bin"
Environment="PYTHONUNBUFFERED=1"
Environment="PRAVAHA_API_KEY=sk-pravaha-prod-8a9b7c6d5e4f3a2b1c"
Environment="PRAVAHA_GPU_MEM_UTIL=0.88"
Environment="SHA256_AUDIT_LOG_DIR=/var/lib/pravaha/audit"

ExecStart=/opt/pravaha/venv/bin/pravaha serve meta-llama/Llama-3-8B-Instruct \
    --config /etc/pravaha/production.yaml \
    --quantize 4bit \
    --swarm \
    --port 8000

Restart=always
RestartSec=10s
LimitNOFILE=65536
LimitNPROC=4096
LimitMEMLOCK=infinity
TasksMax=10000

# Security Boundaries
ProtectSystem=full
ProtectHome=true
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

### 3. Service Management Operations

```bash
# Reload systemd manager configuration
sudo systemctl daemon-reload

# Enable service launch on system boot
sudo systemctl enable pravaha.service

# Start Pravāha engine service
sudo systemctl start pravaha.service

# Monitor live journal logs
sudo journalctl -u pravaha.service -f --output=cat
```

---

## 4. Docker Containerization

Pravāha v3.3 provides optimized multi-stage Containerfile definitions for both CPU testing and CUDA GPU acceleration.

### Production CUDA Containerfile (`docker/Dockerfile.cuda`)

```dockerfile
# Multi-Stage Production Build for Pravāha v3.3 CUDA
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install build system dependencies & Rust toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

# Copy source directory
COPY . .

# Build virtual environment & Rust FFI bindings
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel maturin
RUN pip install --no-cache-dir -e ".[gpu]"

WORKDIR /build/rust
RUN maturin develop --release

# Final Lightweight Runtime Image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    ca-certificates \
    curl \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build /app

WORKDIR /app

# Non-root security user setup
RUN useradd -m -u 10001 pravaha \
    && mkdir -p /app/data /app/audit \
    && chown -R pravaha:pravaha /app

USER pravaha

EXPOSE 8000

ENTRYPOINT ["pravaha", "serve"]
CMD ["meta-llama/Llama-3-8B-Instruct", "--config", "configs/default.yaml", "--quantize", "4bit", "--swarm"]
```

### Docker Run Execution Command

```bash
docker run -d \
  --name pravaha-engine \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -e PRAVAHA_API_KEY="sk-pravaha-live-secret" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /mnt/models:/root/.cache/huggingface \
  -v /var/pravaha/audit:/app/audit \
  pravaha:cuda
```

---

## 5. Docker Compose Enterprise Stack

The full production suite bundles Pravāha v3.3, Prometheus metrics collection, Grafana visualization, and a local Redis rate-limiting node.

```yaml
version: '3.8'

services:
  pravaha:
    build:
      context: ..
      dockerfile: docker/Dockerfile.cuda
    image: pravaha:3.3-cuda
    container_name: pravaha-core
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - PRAVAHA_MODEL_PATH=meta-llama/Llama-3-8B-Instruct
      - PRAVAHA_QUANTIZATION=4bit
      - PRAVAHA_API_KEY=${PRAVAHA_API_KEY:-sk-pravaha-dev-key}
      - PRAVAHA_GPU_MEM_UTIL=0.85
      - SHA256_AUDIT_LOG_DIR=/app/audit
    volumes:
      - pravaha_data:/app/data
      - pravaha_audit:/app/audit
      - /var/run/docker.sock:/var/run/docker.sock
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/ready"]
      interval: 15s
      timeout: 5s
      retries: 3
      start_period: 40s

  prometheus:
    image: prom/prometheus:v2.48.0
    container_name: pravaha-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=15d'

  grafana:
    image: grafana/grafana:10.2.0
    container_name: pravaha-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=pravaha_secure_pass
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro

volumes:
  pravaha_data:
  pravaha_audit:
  prometheus_data:
  grafana_data:
```

---

## 6. Kubernetes Production Deployment

For enterprise orchestration, Pravāha is deployed on Kubernetes clusters using custom GPU scheduling primitives and custom Prometheus metrics autoscaling.

### 1. Namespace, Secret & ConfigMap (`k8s/base-config.yaml`)

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: pravaha-system
---
apiVersion: v1
kind: Secret
metadata:
  name: pravaha-secrets
  namespace: pravaha-system
type: Opaque
stringData:
  PRAVAHA_API_KEY: "sk-pravaha-prod-k8s-secret-9988"
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: pravaha-config
  namespace: pravaha-system
data:
  production.yaml: |
    engine:
      model_path: "meta-llama/Llama-3-8B-Instruct"
      quantization: "4bit"
      device: "cuda"
    scheduler:
      max_num_seqs: 256
      continuous_batching: true
    kv_cache:
      gpu_memory_utilization: 0.85
      block_size: 16
    security:
      enable_auth: true
      enable_rbac: true
    observability:
      enable_prometheus: true
```

### 2. Deployment Manifest (`k8s/deployment.yaml`)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pravaha-engine
  namespace: pravaha-system
  labels:
    app.kubernetes.io/name: pravaha
    app.kubernetes.io/part-of: pravaha-platform
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pravaha-engine
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: pravaha-engine
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: pravaha-core
          image: registry.enterprise.io/pravaha/core:v3.3
          imagePullPolicy: IfNotPresent
          ports:
            - name: http-api
              containerPort: 8000
          envFrom:
            - secretRef:
                name: pravaha-secrets
          volumeMounts:
            - name: config-volume
              mountPath: /etc/pravaha
            - name: audit-volume
              mountPath: /app/audit
            - name: dshm
              mountPath: /dev/shm
          resources:
            limits:
              cpu: "12"
              memory: "32Gi"
              nvidia.com/gpu: "1"
            requests:
              cpu: "4"
              memory: "16Gi"
              nvidia.com/gpu: "1"
          livenessProbe:
            httpGet:
              path: /health
              port: http-api
            initialDelaySeconds: 45
            periodSeconds: 15
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health/ready
              port: http-api
            initialDelaySeconds: 30
            periodSeconds: 5
            successThreshold: 1
      volumes:
        - name: config-volume
          configMap:
            name: pravaha-config
        - name: audit-volume
          persistentVolumeClaim:
            claimName: pravaha-audit-pvc
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 8Gi
```

### 3. Ingress Definition (`k8s/ingress.yaml`)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pravaha-ingress
  namespace: pravaha-system
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
    nginx.ingress.kubernetes.io/websocket-services: "pravaha-service"
spec:
  tls:
    - hosts:
        - pravaha.internal.company.com
      secretName: pravaha-tls-cert
  rules:
    - host: pravaha.internal.company.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: pravaha-service
                port:
                  number: 8000
```

---

## 7. High Availability & Circuit Breaking

Pravāha v3.3 contains a built-in `CircuitBreaker` (`pravaha/engine/circuit_breaker.py`) designed to insulate the server against system cascading failures.

### Circuit Breaker Configuration & Mechanics

```python
class CircuitBreakerConfig:
    failure_threshold: int = 5         # Failures allowed before tripping
    recovery_time_seconds: float = 30.0 # Time window in OPEN state
    half_open_max_calls: int = 3       # Verification probes in HALF_OPEN state
```

### State Transitions

```
        ┌─────────────────────────────────────────────────────────┐
        │                                                         │
        ▼                                                         │
  ┌──────────┐      Failures >= Threshold       ┌──────────┐      │ Probe Success
  │  CLOSED  │ ───────────────────────────────► │   OPEN   │      │
  └────▲─────┘                                  └────┬─────┘      │
       │                                             │            │
       │ Recovery Success                            │ Timeout    │
       │                                             ▼            │
       └───────────────────────────────────── ┌───────────┐       │
                                              │ HALF_OPEN │ ──────┘
                                              └───────────┘
                                                    │
                                                    │ Probe Failure
                                                    ▼
                                              (Re-enters OPEN)
```

In a multi-instance Kubernetes deployment, tripped circuit breakers automatically signal unreadiness to the `/health/ready` probe, causing Kubernetes service proxies to divert traffic away from the recovering node until state stability is re-established.

---

## 8. Security Hardening Checklist

1. **Enable Bearer Authorization**: Set `PRAVAHA_API_KEY` to a cryptographically strong string (minimum 32 characters). Never leave this value empty in production.
2. **Enforce Role Boundaries**: Pass `X-User-Role` headers at the API Gateway level based on validated JWT token claims (`ADMIN` for control operations, `OPERATOR` for swarm runs, `USER` for standard completions).
3. **Isolate Code Execution**: Ensure `DockerSandbox` is active with strict memory limits (`256m`) and CPU quotas (`1.0` core) for all ReAct code-executing agents.
4. **Persist Cryptographic Audit Ledger**: Mount an external persistent volume to `/app/audit` to guarantee that append-only `SHA256AuditTrail` logs survive container restarts.
5. **Disable Default Root Containers**: Always execute worker processes under non-root system accounts (`UID 10001`).

---

## 9. Disaster Recovery & Zero-Downtime Upgrades

### Rolling Model Deployment

When updating model weights or configuration definitions:

1. Update the Kubernetes Deployment image tag or ConfigMap.
2. The `RollingUpdate` strategy initializes new pods alongside old instances (`maxSurge: 1`).
3. Readiness probes (`/health/ready`) keep new pods out of service rotation until the heavy LLM weights are fully loaded into GPU memory.
4. Once the readiness probe returns HTTP 200 OK, traffic transitions smoothly without dropped connections.

### Backup Strategy for Vector Stores & Memory

- **SQLite Swarm Memory**: Execute `VACUUM INTO '/backup/swarm_memory.db'` periodically to capture atomic snapshots of `MemoryStore`.
- **FAISS Vector Indexes**: Copy `.faiss` index files alongside metadata JSON blocks during scheduled low-traffic maintenance windows.

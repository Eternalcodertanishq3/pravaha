# Getting Started with Pravāha v3

## Prerequisites

- Python 3.11+
- Rust toolchain (for the block allocator)
- CUDA 12+ (optional, for GPU inference)

## Installation

### From source (recommended)

```bash
git clone https://github.com/pravaha/pravaha.git
cd pravaha
pip install -e ".[all]"
```

### Minimal install (CPU only)

```bash
pip install -e "."
```

### With GPU support

```bash
pip install -e ".[gpu]"
```

### With all extras

```bash
pip install -e ".[all]"  # Includes: gpu, rag, tui, dev
```

## First Run

### Serve a model

```bash
pravaha serve gpt2
```

This starts the engine on `http://localhost:8000` with:
- OpenAI-compatible API (`/v1/completions`, `/v1/chat/completions`)
- Health check (`/health`)
- Model list (`/v1/models`)

### Interactive chat

```bash
pravaha chat --server http://localhost:8000
```

### With the TUI dashboard

```bash
pravaha serve gpt2 --tui
```

### With quantization (for large models)

```bash
pravaha serve meta-llama/Llama-3-8B --quantize 4bit --tui
```

### With the full swarm

```bash
pravaha serve gpt2 --swarm --self-heal --tui
```

## Configuration

Create a custom config:

```yaml
# my_config.yaml
model:
  model_path: meta-llama/Llama-3-8B
  quantization: 4bit
  device: auto

swarm:
  enabled: true
  max_iterations: 3

rag:
  enabled: true
```

Use it:

```bash
pravaha serve gpt2 --config my_config.yaml
```

## Next Steps

- [CLI Reference](cli.md) — All commands and options
- [API Reference](api.md) — OpenAI-compatible endpoints
- [Swarm Guide](swarm.md) — How the 32-agent system works
- [Configuration Guide](configuration.md) — All config options

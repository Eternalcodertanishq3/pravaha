# Pravāha — प्रवाह

**A vLLM-inspired LLM inference engine with continuous batching and PagedAttention.**

Pravāha means "flow/stream" in Sanskrit, symbolizing continuous batching and token streaming.

## Features (Phase 1 — Baseline)

- ✅ HuggingFace model loading (GPT-2, Llama, Mistral)
- ✅ Configurable dtype (FP16/BF16/FP32)
- ✅ Streaming token generation
- ✅ Sampling pipeline (temperature, top-k, top-p, repetition penalty)
- ✅ GPU memory estimation and monitoring
- ✅ YAML-based configuration

## Roadmap

- ✅ Phase 2: Naive KV-Cache + Streaming Generation
- 🔲 Phase 3: Continuous Batching Scheduler
- 🔲 Phase 4: Paged KV-Cache + BlockAllocator
- 🔲 Phase 5: INT8/INT4 Quantization (GPTQ/AWQ)
- 🔲 Phase 6: API Server + Streaming
- 🔲 Phase 7: Metrics + Profiler
- 🔲 Phase 8: FlashAttention + Speculative Decoding

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run
python -c "
from pravaha.engine import PravahaEngine
engine = PravahaEngine()
for token in engine.generate('Once upon a time', max_new_tokens=50, temperature=0.8):
    print(token, end='', flush=True)
print()
"

# Tests
python -m pytest tests/ -v             # Fast tests only
python -m pytest tests/ -v --run-slow  # All tests (downloads models)
```

## Project Structure

```
pravaha/
├── config.py           # Pydantic configuration system
├── engine.py           # Top-level inference orchestrator
├── models/
│   ├── loader.py       # HuggingFace model loading
│   ├── model_config.py # Architecture detection
│   └── weights.py      # Weight loading utilities
├── tokenizer/
│   └── tokenizer.py    # HuggingFace tokenizer wrapper
├── decoder/
│   ├── decoder.py      # Autoregressive decode loop
│   └── sampling.py     # Sampling strategies
├── scheduler/
│   └── request.py      # Request/sequence data structures
├── kv_cache/           # (Phase 2-4)
├── quantization/       # (Phase 5)
├── server/             # (Phase 6)
└── metrics/            # (Phase 7)
```

## License

MIT

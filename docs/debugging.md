# Debugging & Profiling

## Request Replay

Re-execute a previously recorded request:

```bash
pravaha debug replay <request-id>
# or via API
curl -X POST http://localhost:8000/v1/debug/replay \
  -d '{"request_id": "req-abc123"}'
```

## Token-Level Debugging

Step through inference at any token position:

```bash
pravaha debug step <request-id> --pos 42
```

Shows:
- Token text and ID at that position
- Top-10 candidate tokens with probabilities
- Sampling decision (which token was chosen and why)

## Full Trace Export

Export the complete decision trace for a request:

```bash
pravaha debug trace <request-id>
```

Returns JSON with every token's logits, sampling params, and timing.

## Logit Inspection

View raw logits at a specific position:

```bash
pravaha debug logits <request-id> <token-pos>
```

## Performance Profiling

### Self-Benchmark

Runs automatically on startup. View results:

```bash
pravaha bench --model gpt2 --runs 5
```

### Prometheus Metrics

Exported at `/metrics`:

- `pravaha_requests_total` — Total requests
- `pravaha_tokens_generated` — Total tokens
- `pravaha_ttft_seconds` — Time to first token histogram
- `pravaha_vram_bytes` — GPU memory usage

### Grafana Dashboard

Pre-configured at `http://localhost:3000` when using Docker Compose.

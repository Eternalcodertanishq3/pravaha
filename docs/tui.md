# TUI Dashboard Guide

## Overview

Pravāha includes a premium terminal dashboard built with [Textual](https://textual.textualize.io/).

## Launch

```bash
pravaha serve gpt2 --tui
```

## Panels

### 1. Header Panel
Shows model name, quantization mode, GPU status, and current time.

### 2. Chat Panel
Full chat interface with streaming token rendering. Type messages and see responses appear token-by-token.

### 3. Metrics Panel
Real-time ASCII gauge bars for:
- **Throughput** (tokens/second)
- **TTFT** (time to first token) — p50 and p99
- **VRAM** usage vs total
- **Requests served** and **total tokens**

### 4. Queue Panel
Scheduler request queue utilization bar showing active vs total slots.

### 5. Swarm Panel
32-agent status grid with status indicators:
- `●` Active (currently executing)
- `◆` Auditing (in audit phase)
- `○` Idle (waiting)

### 6. Audit Panel
Shows self-healing loop status: current auditor, iteration count, issues found, patches applied.

### 7. RAG Panel
Document store status: document count, chunk count, last query time.

### 8. Log Panel
Color-coded structured log viewer:
- Blue — INFO
- Yellow — WARN
- Red — ERROR
- Orange — AUDIT

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `q` | Quit |
| `d` | Toggle dark mode |
| `r` | Refresh all panels |

## Theme

The TUI uses a dark green terminal aesthetic defined in `pravaha/tui/pravaha.tcss`. Customizable via standard Textual CSS.

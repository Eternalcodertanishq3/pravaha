# Swarm System Guide

## Overview

The Pravāha swarm is a 32-agent system that collaborates to produce, audit, and self-heal outputs. Every agent is a specialized LLM prompt expert with its own system prompt, priority level, and token budget.

## How It Works

1. **RouterAgent** classifies the input task (code, research, writing, etc.)
2. **PlannerAgent** decomposes it into subtasks
3. **Worker agents** execute subtasks in sequence
4. **Audit agents** scan the output for issues
5. **PatchApplierAgent** fixes any issues found
6. **OutputVerifierAgent** scores task satisfaction (0-100)
7. If score < threshold, the audit loop repeats (up to 3 times)

## Agent Categories

### Workers (20)

Workers produce output. They run in the order defined by the pipeline.

Each worker has:
- **role** — unique identifier
- **priority** — execution order (2=first, 0=last)
- **system_prompt** — expert instructions
- **max_tokens** — token budget
- **temperature** — generation creativity

### Auditors (12)

Auditors scan output for problems. They run after workers complete.

Each auditor returns:
- **issues** — list of problems found
- **confidence** — how sure the audit is
- **clean** — whether the output passed

## Built-in Pipelines

| Pipeline | Workers | Auditors | Best For |
|---|---|---|---|
| `plan-execute-audit` | Planner → Researcher → Coder | Syntax, Security, Logic, EdgeCase, Verifier | General coding |
| `research-summarize` | Researcher → Reasoning → Summarizer | Hallucination, Consistency, Verifier | Research |
| `code-review` | Coder → Debugger → Critic → Refiner | Syntax, Type, Security, Perf, Test, Verifier | Code quality |
| `creative-write` | Narrator → Expander → Refiner | Consistency, Verifier | Creative writing |
| `extract-classify` | Extractor → Classifier → Validator | Verifier | Data extraction |

## Custom Pipelines

Define pipelines in `configs/swarm_default.yaml`:

```yaml
swarm:
  pipelines:
    my-custom:
      workers: [researcher, coder, critic, refiner]
      auditors: [syntax_audit, security_audit, output_verifier]
```

## SharedContext

All agents communicate through `SharedContext` — no side-channel state sharing:

- `task` — original user task
- `output` — current working output
- `code` — generated code
- `research` — gathered research
- `feedback` — critic feedback
- `agent_outputs` — dict of all agent results
- `audit_reports` — list of audit iteration reports

## Enabling Swarm

```bash
pravaha serve gpt2 --swarm --self-heal
```

Or in config:

```yaml
swarm:
  enabled: true
  max_iterations: 3
  min_score: 70.0
```

## API Usage

```bash
curl -X POST http://localhost:8000/v1/swarm/run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python REST API", "pipeline": "code-review"}'
```

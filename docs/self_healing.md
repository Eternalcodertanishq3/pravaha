# Self-Healing Pipeline

## Overview

Pravāha's self-healing pipeline is the core innovation. After worker agents produce output, a pipeline of 12 audit agents scans for issues. If problems are found, the PatchApplier fixes them and the output is re-audited — up to 3 iterations.

## The Audit Loop

```
Worker Output
    ↓
┌─ Audit Iteration 1 ──────────────┐
│  SyntaxAudit     → [clean]       │
│  TypeSafety      → [2 issues]    │
│  SecurityAudit   → [1 vuln]      │
│  LogicFlaw       → [clean]       │
│  ConsistencyGuard → [clean]      │
│  HallucinationHunter → [clean]   │
│  EdgeCaseHunter  → [3 cases]     │
│  PerfProfiler    → [clean]       │
│  OutputVerifier  → score: 45/100 │
│                                   │
│  → 6 issues found                 │
│  → PatchApplier: 5 patched       │
└───────────────────────────────────┘
    ↓
┌─ Audit Iteration 2 ──────────────┐
│  SyntaxAudit     → [clean]       │
│  TypeSafety      → [clean]       │
│  SecurityAudit   → [clean]       │
│  OutputVerifier  → score: 82/100 │
│                                   │
│  → 0 issues, score ≥ 70          │
│  → PASS ✓                        │
└───────────────────────────────────┘
```

## Audit Agents

### Phase 1: Static Analysis (Free, No LLM Cost)

- **SyntaxAuditAgent** — Python AST parsing catches syntax errors instantly
- **SecurityAuditAgent** — 8 regex patterns catch hardcoded secrets, eval(), pickle, etc.

### Phase 2: LLM Deep Analysis

- **TypeSafetyAgent** — Missing annotations, None-dereference
- **LogicFlawAgent** — Contradictions, infinite loops, off-by-one
- **HallucinationHunterAgent** — Fabricated facts, non-existent APIs
- **ConsistencyGuardAgent** — Cross-agent contradictions
- **EdgeCaseHunterAgent** — Empty inputs, overflow, race conditions
- **PerformanceProfilerAgent** — O(n²), N+1 queries, blocking I/O

### Phase 3: Scoring

- **OutputVerifierAgent** — Scores 0-100 on task satisfaction

### Phase 4: Fix Application

- **PatchApplierAgent** — Surgical minimal patches with `# PATCHED:` comments

## Output-Type-Aware Selection

The audit pipeline adapts based on output type:

| Output Type | Auditors Used |
|---|---|
| **Code** | All 9 auditors |
| **Text** | Logic, Hallucination, Consistency, Verifier |
| **Analysis** | Logic, Hallucination, Consistency, EdgeCase, Verifier |

## Configuration

```yaml
swarm:
  max_iterations: 3    # Maximum audit loops
  min_score: 70.0      # Minimum pass score (0-100)
```

## AuditResult

Every audit run returns:

```python
@dataclass
class AuditResult:
    final_output: str           # Patched output
    issues_found: list[dict]    # All issues across iterations
    patches_applied: list[str]  # Applied fixes
    iterations: int             # How many loops ran
    satisfaction_score: float   # Final verifier score
    passed: bool                # Whether threshold was met
```

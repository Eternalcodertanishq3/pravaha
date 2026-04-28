# PRAVAHA v3.3 — Agent & Swarm Reference Guide

> **Version**: 3.3.0
> **Agents**: 51 (20 workers + 13 auditors + 10 security + 9 design)
> **Tools**: 12
> **Pipelines**: 8 pre-built + custom

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Agent Categories](#agent-categories)
3. [Worker Agents (20)](#worker-agents-20)
4. [Audit Agents (13)](#audit-agents-13)
5. [Security Agents (10)](#security-agents-10)
6. [Design Agents (9)](#design-agents-9)
7. [Tool Suite (12)](#tool-suite-12)
8. [Pipelines (8)](#pipelines-8)
9. [ReAct Loop](#react-loop)
10. [SharedContext](#sharedcontext)
11. [Creating Custom Agents](#creating-custom-agents)
12. [Creating Custom Pipelines](#creating-custom-pipelines)
13. [Configuration](#configuration)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│             Pravaha v3.3 Engine                  │
│  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  Decoder   │  │Scheduler │  │ KV Cache     │  │
│  └─────┬─────┘  └────┬─────┘  └──────┬───────┘  │
│        └──────────────┼───────────────┘          │
│                       │                          │
│              ┌────────▼────────┐                 │
│              │    Event Bus    │                 │
│              └────────┬────────┘                 │
│                       │                          │
│  ┌────────────────────▼─────────────────────┐   │
│  │          Swarm Orchestrator               │   │
│  │  ┌──────┐  ┌──────┐  ┌──────┐            │   │
│  │  │Worker│  │Audit │  │Secur │  ┌──────┐  │   │
│  │  │ (20) │  │ (13) │  │ (10) │  │Design│  │   │
│  │  └──────┘  └──────┘  └──────┘  │ (9)  │  │   │
│  │                                └──────┘  │   │
│  └──────────────────────────────────────────┘   │
│                       │                          │
│  ┌────────────────────▼─────────────────────┐   │
│  │   Tool Registry (12) + Memory Store      │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### Key Concepts

- **BaseAgent**: Abstract class with ReAct loop, tool access, persistent memory
- **SharedContext**: Mutable state shared between agents in a pipeline
- **ToolRegistry**: Central registry of 12 tools agents can invoke
- **EventBus**: Pub/sub system for engine↔swarm↔TUI communication
- **SwarmProfiler**: Per-agent performance tracking (mean/p95 duration, token efficiency)
- **PipelineDAG**: DAG-based execution for parallel agent steps

---

## Agent Categories

| Category   | Count | Purpose                                    |
|-----------|------:|--------------------------------------------|
| Workers   |    20 | Core task execution — coding, planning, research |
| Auditors  |    13 | Quality assurance — syntax, logic, output verification |
| Security  |    10 | Vulnerability scanning — injection, auth, crypto |
| Design    |     9 | UI/UX — layouts, components, accessibility |
| **Total** |**51** | **Full swarm ecosystem**                   |

---

## Worker Agents (20)

| Agent          | Role          | Priority | Temperature | Tools | Description |
|---------------|--------------|----------|-------------|-------|-------------|
| PlannerAgent   | `planner`    | 10       | 0.5         | ✓     | Decompose tasks into actionable steps |
| ResearcherAgent| `researcher` | 9        | 0.5         | ✓     | Gather information from tools and context |
| CoderAgent     | `coder`      | 8        | 0.2         | ✓     | Write and modify code |
| DebuggerAgent  | `debugger`   | 8        | 0.2         | ✓     | Find and fix bugs |
| ReasoningAgent | `reasoning`  | 7        | 0.5         | —     | Chain-of-thought reasoning |
| CriticAgent    | `critic`     | 6        | 0.4         | —     | Evaluate quality and suggest improvements |
| RefinerAgent   | `refiner`    | 5        | 0.3         | —     | Polish and refine outputs |
| SummarizerAgent| `summarizer` | 5        | 0.3         | —     | Condense information |
| NarratorAgent  | `narrator`   | 4        | 0.8         | —     | Write fluent prose |
| ExpanderAgent  | `expander`   | 4        | 0.8         | —     | Elaborate on topics |
| ExtractorAgent | `extractor`  | 4        | 0.2         | —     | Extract structured data |
| ClassifierAgent| `classifier` | 9        | 0.2         | —     | Classify task types |
| RouterAgent    | `router`     | 9        | 0.2         | —     | Route tasks to pipelines |
| TranslatorAgent| `translator` | 3        | 0.3         | —     | Language translation |
| EnsembleAgent  | `ensemble`   | 3        | 0.5         | —     | Combine multiple outputs |
| MergerAgent    | `merger`     | 3        | 0.3         | —     | Merge results from parallel agents |
| JudgeAgent     | `judge`      | 6        | 0.3         | —     | Score and rank outputs |
| MemoryAgent    | `memory`     | 7        | 0.3         | ✓     | Retrieve and store agent memories |
| ToolAgent      | `tool`       | 7        | 0.3         | ✓     | Execute tool operations |
| ValidatorAgent | `validator`  | 6        | 0.2         | ✓     | Validate outputs against requirements |

---

## Audit Agents (13)

### Phase A: Static Auditors (instant, no LLM)

| Agent              | Role               | Description |
|-------------------|-------------------|-------------|
| SyntaxAudit       | `syntax_audit`    | Regex-based syntax checking |
| TypeSafety        | `type_safety`     | Type annotation validation |
| SecurityAudit     | `security_audit`  | Common vulnerability patterns |
| InjectionScanner  | `injection_scanner`| SQL/XSS/command injection |
| CryptoAudit       | `crypto_audit`    | Cryptographic weakness detection |
| SecretsScanner    | `secrets_scanner` | Hardcoded secrets/API keys |
| PrivilegeAudit    | `privilege_audit` | Privilege escalation risks |
| NetworkSecurity   | `network_security`| Network exposure patterns |
| Compliance        | `compliance`      | Regulatory compliance checks |
| AuthAudit         | `auth_audit`      | Authentication flow review |
| DependencyAudit   | `dependency_audit`| Known vulnerable dependencies |
| APISecurity       | `api_security`    | API security best practices |

### Phase B: LLM Auditors (deep analysis)

| Agent              | Role                  | Description |
|-------------------|----------------------|-------------|
| LogicFlaw         | `logic_flaw`         | Logical error detection |
| HallucinationHunter| `hallucination_hunter`| Fact-checking agent outputs |
| ConsistencyGuard  | `consistency_guard`  | Cross-output consistency |
| EdgeCaseHunter    | `edge_case_hunter`   | Edge case identification |
| PerformanceProfiler| `performance_profiler`| Performance bottleneck analysis |
| OutputVerifier    | `output_verifier`    | Final output quality gate |

---

## Security Agents (10)

All 10 security agents are also listed under audit agents above, but they
form a dedicated security sub-swarm that can be activated for
`secure-code-review` and `full-secure-design` pipelines.

---

## Design Agents (9)

| Agent              | Role                | Description |
|-------------------|---------------------|-------------|
| UIDesigner        | `ui_designer`       | Design user interfaces |
| ComponentBuilder  | `component_builder` | Build reusable components |
| LayoutDesigner    | `layout_designer`   | Create responsive layouts |
| StyleDesigner     | `style_designer`    | Define style systems |
| A11yAuditor       | `accessibility_auditor` | WCAG compliance |
| UXReviewer        | `ux_reviewer`       | User experience review |
| DesignCritic      | `design_critic`     | Design quality review |
| PrototypeBuilder  | `prototype_builder` | Rapid prototyping |
| DesignSystem      | `design_system`     | Design token management |

---

## Tool Suite (12)

| # | Tool            | Name             | Description |
|---|----------------|------------------|-------------|
| 1 | CodeExecutor   | `execute_code`   | Execute Python code in sandboxed environment |
| 2 | FileReader     | `read_file`      | Read file contents from disk |
| 3 | WebFetcher     | `fetch_url`      | Fetch web page content |
| 4 | SearchTool     | `web_search`     | Web search via DuckDuckGo |
| 5 | ShellRunner    | `run_shell`      | Execute shell commands |
| 6 | PythonRepl     | `python_repl`    | Persistent Python REPL with state |
| 7 | BashTool       | `bash`           | Whitelisted bash with pipes |
| 8 | JsonTool       | `json_query`     | Parse/query/transform JSON |
| 9 | HttpClient     | `http_request`   | Full HTTP client (GET/POST/etc) |
|10 | FileWriter     | `write_file`     | Write files to whitelisted paths |
|11 | GitTool        | `git`            | Safe git: status/diff/log/add/commit |
|12 | Calculator     | `calculate`      | Safe math via AST + optional sympy |

### Tool Security Model

- **CodeExecutor**: Timeout-guarded, no file system access
- **BashTool**: Command whitelist (no `rm`, `dd`, etc.)
- **FileWriter**: Directory and extension whitelist
- **GitTool**: No push/pull/reset (read + commit only)
- **HttpClient**: Timeout-guarded, follows redirects

---

## Pipelines (8)

### plan-execute-audit (default)

```
Planner → Coder → Critic → [syntax_audit, security_audit, output_verifier]
```

General-purpose: plan the task, code it, critique it, audit it.

### research-write

```
Researcher → Expander → Narrator → [hallucination_hunter, output_verifier]
```

Research a topic, expand findings, narrate a report.

### code-review

```
Coder → Debugger → Refiner → [syntax, type_safety, security, edge_case, test_gen]
```

Deep code review with 5 auditors.

### secure-code-review

```
Planner → Coder → Debugger → [syntax, security, injection, auth, crypto, secrets, verifier, patch]
```

Security-hardened review with 8 auditors.

### design-component

```
UIDesigner → LayoutDesigner → StyleDesigner → ComponentBuilder → [a11y, ux, design_critic, verifier]
```

Full design pipeline with accessibility checking.

### full-secure-design

```
Planner → UIDesigner → ComponentBuilder → Coder → Debugger → [8 auditors]
```

End-to-end design + security.

### full-pipeline

```
Classifier → Planner → Researcher → Coder → Critic → Refiner → Merger → [9 auditors]
```

The complete pipeline: classify, plan, research, code, critique, refine, merge, audit.

---

## ReAct Loop

Every agent with tool access uses the **ReAct** (Reason + Act) loop:

```
THOUGHT: What do I need to do? What information do I have?
ACTION:  Which tool should I call? With what args?
OBSERVE: What did the tool return?
THOUGHT: Does this answer my question or do I need more?
...repeat up to max_react_steps (default: 5)...
ANSWER:  Final response to the task.
```

### XML Format

```xml
<thought>I need to check the current code for syntax errors</thought>
<action>execute_code</action>
<args>{"code": "import ast; ast.parse(code)"}</args>

<observation>No errors found</observation>

<thought>The code is syntactically valid, I can provide my answer</thought>
<answer>The code passes syntax validation...</answer>
```

---

## SharedContext

The `SharedContext` dataclass is the mutable state shared across all agents
in a pipeline execution:

| Field                  | Type                    | Purpose |
|-----------------------|-------------------------|---------|
| `task`                | `str`                  | Current task description |
| `plan`                | `str`                  | Generated plan |
| `research`            | `str`                  | Research findings |
| `code`                | `str`                  | Generated code |
| `output`              | `str`                  | Current pipeline output |
| `reasoning`           | `str`                  | Reasoning chain |
| `tests`               | `str`                  | Generated tests |
| `context_summary`     | `str`                  | Compressed context |
| `merged_output`       | `str`                  | Merged multi-agent output |
| `patched_output`      | `str`                  | Post-patch output |
| `feedback`            | `str`                  | Iteration feedback |
| `task_type`           | `str`                  | Classified task type |
| `conversation_history`| `list[dict]`           | Message history |
| `agent_outputs`       | `dict[str, AgentOutput]`| Per-agent results |
| `audit_reports`       | `list[dict]`           | Audit findings |
| `extra`               | `dict[str, Any]`       | Arbitrary metadata |

---

## Creating Custom Agents

```python
from pravaha.swarm.agents.base_agent import BaseAgent, AgentOutput, SharedContext

class MyAgent(BaseAgent):
    role = "my_agent"
    priority = 5
    max_tokens = 2048
    temperature = 0.5
    system_prompt = "You are a specialized agent for..."
    available_tools = ["python_repl", "read_file"]
    max_react_steps = 5

    def can_handle(self, task_type: str) -> bool:
        return task_type in ("my_task_type", "general")

    async def run(self, task, context, engine):
        # Option 1: Use ReAct loop (automatic with tools)
        return await self.run_react(task, context, engine)

        # Option 2: Direct generation (no tools)
        prompt = self.build_prompt(task, context)
        output = await self._generate(prompt, engine)
        return AgentOutput(role=self.role, output=output)
```

### Register your agent:

```python
# In pravaha/swarm/agents/__init__.py
from my_module import MyAgent
ALL_AGENTS["my_agent"] = MyAgent
```

---

## Creating Custom Pipelines

### In swarm_default.yaml:

```yaml
pipelines:
  my-pipeline:
    workers: [planner, my_agent, coder, validator]
    auditors: [syntax_audit, output_verifier]
```

### In code:

```python
orchestrator = SwarmOrchestrator()
results = await orchestrator.execute_with_audit(
    worker_pipeline=["planner", "coder", "validator"],
    task="Build a REST API",
    engine=engine,
    max_iterations=3,
    min_score=70.0,
)
```

### DAG Pipelines (v3.3):

```python
from pravaha.swarm.pipeline_dag import PipelineDAG

dag = PipelineDAG()
dag.add_node("plan", agent_role="planner")
dag.add_node("research", agent_role="researcher")  # parallel with plan
dag.add_node("code", agent_role="coder", dependencies=["plan", "research"])
dag.add_node("audit", agent_role="syntax_audit", dependencies=["code"])

result = await dag.execute(run_agent=orchestrator.execute_agent, context=ctx)
```

---

## Configuration

### Temperature Profiles (v3.3)

```yaml
temperature_profiles:
  creative:
    temperature: 0.8
    agents: [narrator, expander, ui_designer]
  analytical:
    temperature: 0.2
    agents: [coder, debugger, validator]
```

### Token Budgets (v3.3)

```yaml
token_budgets:
  default: 1024
  overrides:
    coder: 4096
    planner: 2048
    classifier: 256
```

---

*Generated for Pravaha v3.3.0 — AI Agentic Orchestration Framework*

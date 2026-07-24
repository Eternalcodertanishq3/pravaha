# Conversation Branching Architecture & Operations

Conversation Branching in Pravāha v3.3 allows developers and interactive users to fork chat histories into Directed Acyclic Graphs (DAGs), enabling non-linear prompt exploration, speculative path execution, and multi-agent hypothesis testing. Built directly atop Pravāha’s **PagedAttention KV-cache** engine and **Rust-powered PrefixTrie**, branching achieves zero-copy memory reuse for shared conversational prefixes.

---

## 1. Executive Overview & Architectural Philosophy

Traditional conversational AI platforms maintain linear context sequences. When a user edits a prior prompt or explores alternative reasoning paths, traditional frameworks either discard the original context or duplicate the entire Key-Value (KV) cache in GPU memory.

Pravāha v3.3 treats conversation history as a **tree of discrete nodes**, where each node represents a user message, assistant response, or tool observation step.

```
                      [ Root Node: System Prompt ]
                                   │
                         [ Node 1: User Request ]
                                   │
                      [ Node 2: Assistant Strategy ]
                                /        \
                               /          \
      [ Branch A: Node 3A (Python) ]   [ Branch B: Node 3B (Rust) ]
                     │                                │
      [ Node 4A: Code Output ]          [ Node 4B: Code Output ]
```

### Key Architectural Benefits
- **Zero-Copy Memory Overhead**: Divergent branches share identical KV-cache memory blocks up to the point of bifurcation.
- **Sub-Millisecond Context Switching**: Switching active conversation branches requires updating token pointer references rather than recomputing attention matrices.
- **Auditable History Graph**: Every branch point, parent pointer, and execution path is recorded for SHA-256 tamper-evident audit log tracing.

---

## 2. DAG Data Model & Session Structures

The branching system is implemented in `pravaha/serving/routes/branches.py` and backed by the core engine's session cache manager.

### Core Data Structures (`SessionNode`, `Branch`, `ConversationTree`)

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

@dataclass
class SessionNode:
    """Represents a single message node within the conversation DAG."""
    node_id: str
    parent_id: Optional[str]
    role: str
    content: str
    created_at: float = field(default_factory=time.time)
    token_count: int = 0
    block_ids: List[int] = field(default_factory=list)

@dataclass
class Branch:
    """Represents a named lineage path through the conversation DAG."""
    branch_id: str
    session_id: str
    fork_node_id: str
    active_head_id: str
    label: str
    created_at: float = field(default_factory=time.time)

class ConversationTree:
    """Manages DAG node linkages, lineage traversal, and active branch checkout."""
    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.nodes: Dict[str, SessionNode] = {}
        self.branches: Dict[str, Branch] = {}
        self.active_branch_id: str = "main"

    def get_lineage(self, node_id: str) -> List[SessionNode]:
        """Traverse backwards from node_id to root to reconstruct message history."""
        lineage = []
        curr = self.nodes.get(node_id)
        while curr:
            lineage.append(curr)
            curr = self.nodes.get(curr.parent_id) if curr.parent_id else None
        return list(reversed(lineage))
```

---

## 3. KV-Cache Optimization & Prefix Sharing Integration

Pravāha’s branching performance is tightly coupled with **PagedAttention** (`pravaha/memory/block_manager.py`) and the **Rust PrefixTrie** (`rust/src/prefix_trie.rs`).

```
                              Prefix Tokens (Shared)
                          ┌───────────────────────────┐
                          │ Block 001  │  Block 002   │
                          └────────────┴──────────────┘
                                  ▲          ▲
                     Ref Count = 2│          │ Ref Count = 2
                                  │          │
                 ┌────────────────┴──────────┴────────────────┐
                 │                                            │
        Branch A KV Blocks                           Branch B KV Blocks
     ┌──────────────────────┐                     ┌──────────────────────┐
     │ Block 003 (Python)   │                     │ Block 004 (Rust)     │
     └──────────────────────┘                     └──────────────────────┘
```

### Prefix Sharing Mechanics
1. **Fork Identification**: When a branch is created at message index $K$, Pravāha identifies the sequence of tokens $T_{0..K}$ representing the shared prefix.
2. **Rust PrefixTrie Lookup**: The engine calls `longest_prefix_match(T_{0..K})` against the Rust trie core to resolve physical block IDs.
3. **Reference Count Increment**: The physical KV memory blocks allocated to $T_{0..K}$ have their reference counters incremented in `BlockManager`.
4. **Copy-on-Write (CoW)**: If Branch A appends new tokens, new blocks (`Block 003`) are allocated from the free list without mutating or duplicating the shared prefix blocks (`Block 001`, `Block 002`).

---

## 4. REST API Reference

All branching endpoints enforce Bearer Authentication and RBAC scoping.

### 4.1 Create Branch
`POST /v1/branch`

Forks a conversation from an existing session at a designated message index.

#### Request Headers
```http
Authorization: Bearer <PRAVAHA_API_KEY>
X-User-Role: user
Content-Type: application/json
```

#### Request JSON Schema
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["session_id", "fork_at"],
  "properties": {
    "session_id": {
      "type": "string",
      "description": "Unique session identifier."
    },
    "fork_at": {
      "type": "integer",
      "minimum": 0,
      "description": "Zero-indexed message index at which to fork."
    },
    "label": {
      "type": "string",
      "default": "unnamed-branch",
      "description": "Human-readable label for the branch."
    }
  }
}
```

#### Request Body Example
```json
{
  "session_id": "sess_production_99",
  "fork_at": 2,
  "label": "refactor-async-await"
}
```

#### Response (`201 Created`)
```json
{
  "branch_id": "br_refactor_async_await_4f8a",
  "session_id": "sess_production_99",
  "fork_node_id": "node_msg_002",
  "active_head_id": "node_msg_002",
  "label": "refactor-async-await",
  "created_at": 1774343050,
  "shared_prefix_tokens": 142,
  "blocks_reused": 9
}
```

---

### 4.2 List Session Branches
`GET /v1/branch/{session_id}`

Retrieves the complete branch tree structure for a given session.

#### Response (`200 OK`)
```json
{
  "session_id": "sess_production_99",
  "active_branch_id": "br_refactor_async_await_4f8a",
  "branches": [
    {
      "branch_id": "main",
      "label": "Default Branch",
      "fork_node_id": "root",
      "head_node_id": "node_msg_005",
      "node_count": 6
    },
    {
      "branch_id": "br_refactor_async_await_4f8a",
      "label": "refactor-async-await",
      "fork_node_id": "node_msg_002",
      "head_node_id": "node_msg_004_b",
      "node_count": 4
    }
  ]
}
```

---

### 4.3 Checkout Branch
`POST /v1/branch/{branch_id}/checkout`

Sets the specified branch as active for the session. Subsequent chat completions using `session_id` will append tokens to this branch head.

#### Response (`200 OK`)
```json
{
  "status": "success",
  "session_id": "sess_production_99",
  "active_branch_id": "br_refactor_async_await_4f8a",
  "active_head_id": "node_msg_004_b",
  "history_length": 4
}
```

---

### 4.4 Delete Branch
`DELETE /v1/branch/{branch_id}`

Deletes a branch lineage. If blocks assigned to the branch are no longer referenced by any other active branch, `BlockManager` decrements their reference counts and releases unreferenced blocks to the free list.

#### Response (`200 OK`)
```json
{
  "status": "deleted",
  "branch_id": "br_refactor_async_await_4f8a",
  "freed_blocks": 3,
  "retained_shared_blocks": 9
}
```

---

## 5. CLI & Interactive REPL Workflows

The Pravāha CLI (`pravaha chat`) provides seamless terminal control over conversation branching.

### 5.1 Launching Chat with a Specific Branch
```bash
pravaha chat --model meta-llama/Llama-3-8B --branch br_refactor_async_await_4f8a
```

### 5.2 Slash Commands in REPL

| Command | Arguments | Description |
| :--- | :--- | :--- |
| `/branch` | `[label]` | Forks the current conversation at the latest user message. |
| `/branch list` | None | Displays an ASCII tree rendering of all branches in the session. |
| `/branch checkout` | `<branch_id>` | Switches active context to the target branch. |
| `/branch diff` | `<id_1> <id_2>` | Compares output messages between two branch heads. |

#### Example Terminal Output (`/branch list`)
```
Conversation Tree for session: sess_production_99
├── main (Head: node_msg_005)
│   └── 001: System Prompt
│   └── 002: User: Write auth module
│   └── 003: Assistant: Sync auth code
└── [ACTIVE] refactor-async-await (ID: br_refactor_async_await_4f8a)
    └── 002: User: Write auth module
    └── 004_b: Assistant: Async auth code using HMAC-SHA256
```

---

## 6. Session Lifecycle & Multi-User State Synchronization

When multiple users or agent swarm threads interact with the same session DAG concurrently, Pravāha guarantees thread safety and memory consistency.

```
               Async HTTP / Swarm Requests
              ┌───────────────────────────┐
              │ Request 1 │   Request 2   │
              └─────┬─────────────┬───────┘
                    │             │
                    ▼             ▼
       ┌─────────────────────────────────────────┐
       │ SessionLockManager (asyncio.Lock)       │
       │ Enforces per-session node mutation lock │
       └────────────────────┬────────────────────┘
                            │
                            ▼
       ┌─────────────────────────────────────────┐
       │ Atomic Block Reference Update (Rust)    │
       └─────────────────────────────────────────┘
```

1. **`SessionLockManager`**: Implements fine-grained `asyncio.Lock` structures indexed by `session_id`. Modifying a DAG node or attaching a branch requires holding the session lock.
2. **Atomic Rust Reference Counting**: Reference counter updates in `allocator.rs` use atomic operations (`AtomicUsize`), ensuring cross-thread safety when continuous batching reads shared blocks while a branch is being modified.

---

## 7. Garbage Collection & Memory Management

To prevent memory leaks during heavy speculative branching, Pravāha enforces strict garbage collection rules:

### Block Deallocation Algorithm
When a branch is deleted or expires:
```python
def release_branch(branch: Branch, tree: ConversationTree, block_manager: BlockManager):
    node = tree.nodes.get(branch.head_node_id)
    while node:
        for block_id in node.block_ids:
            # Decrement atomic reference counter
            ref_count = block_manager.decrement_ref(block_id)
            if ref_count == 0:
                # Return block to free pool
                block_manager.free_block(block_id)
        
        # Stop traversing if parent is referenced by another active branch
        if is_node_referenced_by_other_branch(node.parent_id, branch.branch_id, tree):
            break
        node = tree.nodes.get(node.parent_id)
```

### Operational Bounds
- **Maximum Depth**: Default 64 nodes per lineage.
- **Maximum Branches per Session**: Default 128 branches.
- **Session TTL**: Unchecked out branches expire after 3600 seconds (configurable via `cache.session_ttl_seconds`).

---

## 8. Multi-Agent Speculative Path Evaluation

Conversation branching serves as a foundation for autonomous agent speculative path evaluation in the Pravāha Swarm.

```
                        [ Initial Prompt Node ]
                                   │
                      ┌────────────┴────────────┐
                      ▼                         ▼
             [ Branch A: Coder 1 ]     [ Branch B: Coder 2 ]
                      │                         │
             [ Audit Score: 62 ]       [ Audit Score: 88 ]
                      │                         │
                      └────────────┬────────────┘
                                   ▼
                       [ Prune Branch A & Commit B ]
```

### Execution Strategy
1. **Parallel Forking**: The Swarm Orchestrator forks the conversation into $N$ parallel branches.
2. **Independent ReAct Loops**: Separate worker agents execute ReAct reasoning loops concurrently across each branch.
3. **Auditor Scoring**: 12 Auditor agents evaluate the output of each branch head.
4. **Automated Commit & Pruning**: The highest-scoring branch (e.g., score $\ge 70.0$) is committed to the main session line, while failing branches are pruned, releasing their unshared KV-cache blocks back to the allocator.

---

## 9. Python SDK / Programmatic Integration Example

```python
import asyncio
import httpx

API_BASE = "http://localhost:8000/v1"
API_KEY = "pr_live_9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c"

async def explore_branches():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "X-User-Role": "user",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(base_url=API_BASE, headers=headers) as client:
        # Step 1: Initial chat completion
        resp1 = await client.post("/chat/completions", json={
            "model": "meta-llama/Llama-3-8B",
            "messages": [
                {"role": "user", "content": "Write a python function to compute fibonacci."}
            ],
            "session_id": "sess_fib_001"
        })
        print("Main Branch Response:", resp1.json()["choices"][0]["message"]["content"])

        # Step 2: Fork conversation at index 0
        branch_resp = await client.post("/branch", json={
            "session_id": "sess_fib_001",
            "fork_at": 0,
            "label": "iterative-approach"
        })
        branch_id = branch_resp.json()["branch_id"]
        print(f"Created Branch: {branch_id}")

        # Step 3: Checkout new branch
        await client.post(f"/branch/{branch_id}/checkout")

        # Step 4: Request alternative implementation on new branch
        resp2 = await client.post("/chat/completions", json={
            "model": "meta-llama/Llama-3-8B",
            "messages": [
                {"role": "user", "content": "Write an iterative python function to compute fibonacci."}
            ],
            "session_id": "sess_fib_001"
        })
        print("Iterative Branch Response:", resp2.json()["choices"][0]["message"]["content"])

if __name__ == "__main__":
    asyncio.run(explore_branches())
```

---

## 10. Edge Cases & Troubleshooting Guide

### 10.1 Branch State Desynchronization
- **Symptom**: HTTP 404 returned when attempting to checkout a `branch_id`.
- **Cause**: Session expired due to `session_ttl_seconds` threshold being breached, triggering LRU eviction.
- **Resolution**: Increase `cache.session_ttl_seconds` in `configs/default.yaml` or persist DAG nodes to disk.

### 10.2 Memory Pressure During High Branching
- **Symptom**: `MemoryFullError` raised when creating new branches.
- **Cause**: Out-of-memory condition in physical GPU blocks due to unreleased dead branches.
- **Resolution**: Issue explicit `DELETE /v1/branch/{branch_id}` requests or trigger `POST /admin/reload` to flush orphan cache blocks.

### 10.3 Parent Node Deletion Guard
- **Attempting to delete a parent node while child branches are active**: Pravāha safely rejects manual deletion of parent nodes that have active child pointers, returning `409 Conflict`.

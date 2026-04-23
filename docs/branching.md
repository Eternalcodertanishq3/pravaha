# Conversation Branching

Fork, explore, and manage parallel conversation threads.

## Usage

```bash
# Fork at message index 3
curl -X POST http://localhost:8000/v1/branch \
  -d '{"session_id": "abc", "fork_at": 3, "label": "alternative approach"}'

# List branches
curl http://localhost:8000/v1/branch/abc

# Checkout a branch
curl -X POST http://localhost:8000/v1/branch/branch-id-123/checkout

# Delete
curl -X DELETE http://localhost:8000/v1/branch/branch-id-123
```

## CLI

```bash
pravaha chat --branch branch-id-123
```

In the chat REPL, use `/branch` to fork the current conversation.

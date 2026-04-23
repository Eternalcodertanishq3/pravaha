# CLI Reference

Pravāha v3 CLI is built with [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/).

## `pravaha serve`

One-command model serving.

```bash
pravaha serve <MODEL> [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--port`, `-p` | 8000 | Server port |
| `--host` | 0.0.0.0 | Server host |
| `--quantize` | none | `none`, `4bit`, `8bit` |
| `--tui` | false | Launch terminal dashboard |
| `--swarm` | false | Enable 32-agent swarm |
| `--self-heal/--no-self-heal` | true | Self-healing audit loop |
| `--rag` | false | Enable RAG pipeline |
| `--speculative` | false | Speculative decoding |
| `--config`, `-c` | none | YAML config file |
| `--workers` | 1 | Uvicorn worker processes |
| `--benchmark/--no-benchmark` | true | Self-benchmark on startup |

**Examples:**

```bash
pravaha serve gpt2
pravaha serve meta-llama/Llama-3-8B --quantize 4bit --tui
pravaha serve mistral-7b --swarm --self-heal --rag --tui
```

## `pravaha chat`

Interactive REPL with streaming tokens.

```bash
pravaha chat [OPTIONS]
```

| Option | Description |
|---|---|
| `--model`, `-m` | Model to use |
| `--server`, `-s` | Connect to running server URL |
| `--branch` | Continue from branch ID |

**Slash commands:**

| Command | Description |
|---|---|
| `/help` | Show commands |
| `/model` | Show current model |
| `/stats` | Show engine stats |
| `/clear` | Clear conversation |
| `/branch` | Fork conversation |
| `/swarm <task>` | Run swarm pipeline |
| `/rag <query>` | Query RAG store |
| `/audit on\|off` | Toggle audit |
| `/debug` | Show debug info |
| `/quit` | Exit |

## `pravaha bench`

Run benchmarks.

```bash
pravaha bench [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--model`, `-m` | gpt2 | Model to benchmark |
| `--prompt-len` | 128 | Input tokens |
| `--output-len` | 50 | Output tokens |
| `--runs` | 3 | Benchmark runs |
| `--concurrent` | 1 | Concurrent requests |

## `pravaha models`

Model management.

```bash
pravaha models list          # List loaded models
pravaha models info <model>  # Show model details
pravaha models pull <model>  # Download from HuggingFace
pravaha models remove <model> # Remove cached model
```

## `pravaha swarm`

Swarm pipeline management.

```bash
pravaha swarm run "task" [--pipeline NAME] [--max-iter N]
pravaha swarm list-agents
pravaha swarm pipeline <name>
```

## `pravaha rag`

RAG document management.

```bash
pravaha rag ingest <source>     # Ingest file/URL
pravaha rag query "question"    # Query store
pravaha rag list                # List documents
pravaha rag remove <doc-id>     # Remove document
```

## `pravaha debug`

Debug and replay tools.

```bash
pravaha debug replay <request-id>
pravaha debug step <request-id> [--pos N]
pravaha debug trace <request-id>
pravaha debug logits <request-id> <token-pos>
```

## `pravaha plugin`

Plugin management.

```bash
pravaha plugin list
pravaha plugin install <path>
pravaha plugin remove <name>
pravaha plugin info <name>
```

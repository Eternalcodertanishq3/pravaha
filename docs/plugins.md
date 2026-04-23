# Plugin System

## Creating a Plugin

```python
from pravaha.plugins.base_plugin import BasePlugin

class MyPlugin(BasePlugin):
    name = "my-plugin"
    version = "1.0.0"
    description = "Custom plugin"

    def on_load(self) -> None:
        print("Loaded!")

    def on_request(self, request: dict) -> dict:
        return request

    def on_response(self, response: dict) -> dict:
        return response
```

Register via `pyproject.toml` entry points:

```toml
[project.entry-points."pravaha.plugins"]
my-plugin = "my_plugin.plugin:MyPlugin"
```

## CLI

```bash
pravaha plugin list
pravaha plugin install ./my-plugin/
pravaha plugin remove <name>
pravaha plugin info <name>
```

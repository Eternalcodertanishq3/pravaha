"""Quick verification of all v3.3 components."""
import sys
import os
from pathlib import Path
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

tests = []

# 1. EventBus
try:
    from pravaha.engine.event_bus import get_event_bus
    bus = get_event_bus()
    bus.publish("test_verify", {"v": 3.3})
    tests.append(("EventBus", "OK", bus.get_stats()["total_events"]))
except Exception as e:
    tests.append(("EventBus", "FAIL", str(e)))

# 2. LoadBalancer
try:
    from pravaha.engine.load_balancer import AdaptiveLoadBalancer
    lb = AdaptiveLoadBalancer()
    tests.append(("LoadBalancer", "OK", lb.get_device_string()))
except Exception as e:
    tests.append(("LoadBalancer", "FAIL", str(e)))

# 3. HFCompat
try:
    from pravaha.models.hf_compat import HFCompatLayer
    compat = HFCompatLayer()
    tests.append(("HFCompat", "OK", f"flash_attn={compat._flash_attn_available}"))
except Exception as e:
    tests.append(("HFCompat", "FAIL", str(e)))

# 4. Tools (all 12)
try:
    from pravaha.swarm.tools import ToolRegistry
    tr = ToolRegistry.default()
    tool_list = tr.list_tools()
    tests.append(("ToolRegistry", "OK", f"{len(tool_list)} tools: {tool_list}"))
except Exception as e:
    tests.append(("ToolRegistry", "FAIL", str(e)))

# 5. JsonTool
try:
    from pravaha.swarm.tools.json_tool import JsonTool
    j = JsonTool()
    r = j.execute('{"a": {"b": [1, 2, 3]}}', "a.b[1]")
    tests.append(("JsonTool", "OK", r))
except Exception as e:
    tests.append(("JsonTool", "FAIL", str(e)))

# 6. Calculator
try:
    from pravaha.swarm.tools.calculator import Calculator
    c = Calculator()
    r = c.execute("sqrt(144) + 3")
    tests.append(("Calculator", "OK", r["result"]))
except Exception as e:
    tests.append(("Calculator", "FAIL", str(e)))

# 7. PythonRepl
try:
    from pravaha.swarm.tools.python_repl import PythonRepl
    repl = PythonRepl()
    repl.execute("x = 42")
    r = repl.execute("x * 2")
    tests.append(("PythonRepl", "OK", r["stdout"]))
except Exception as e:
    tests.append(("PythonRepl", "FAIL", str(e)))

# 8. SemanticRouter
try:
    from pravaha.router.semantic_router import SemanticRouter
    sr = SemanticRouter()
    route = sr.route("fix the bug in the login function")
    tests.append(("SemanticRouter", "OK", route))
except Exception as e:
    tests.append(("SemanticRouter", "FAIL", str(e)))

# 9. Profiler
try:
    from pravaha.swarm.profiler import SwarmProfiler
    p = SwarmProfiler()
    p.record("coder", 100.0, 200)
    tests.append(("Profiler", "OK", p.get_summary()["total_calls"]))
except Exception as e:
    tests.append(("Profiler", "FAIL", str(e)))

# 10. OutputValidator
try:
    from pravaha.swarm.output_validator import OutputValidator
    v = OutputValidator()
    r = v.validate_text("Hello world", min_length=5)
    tests.append(("OutputValidator", "OK", r.valid))
except Exception as e:
    tests.append(("OutputValidator", "FAIL", str(e)))

# 11. PipelineDAG
try:
    from pravaha.swarm.pipeline_dag import PipelineDAG
    dag = PipelineDAG.from_pipeline_config(["planner", "coder"], ["syntax_audit"])
    order = dag.get_execution_order()
    tests.append(("PipelineDAG", "OK", order))
except Exception as e:
    tests.append(("PipelineDAG", "FAIL", str(e)))

# 12. TUI App
try:
    from pravaha.tui.app import PravahaTUI
    tests.append(("TUI App", "OK", PravahaTUI.TITLE))
except Exception as e:
    tests.append(("TUI App", "FAIL", str(e)))

# 13. Version
try:
    from pravaha import __version__
    tests.append(("Version", "OK", __version__))
except Exception as e:
    tests.append(("Version", "FAIL", str(e)))

# Print results
print("=" * 60)
print("  PRAVAHA v3.3 BUILD VERIFICATION")
print("=" * 60)
passed = 0
failed = 0
for name, status, detail in tests:
    icon = "[OK]" if status == "OK" else "[!!]"
    print(f"  {icon} {name:20s} {status}  {detail}")
    if status == "OK":
        passed += 1
    else:
        failed += 1
print("=" * 60)
print(f"  {passed}/{passed + failed} PASSED")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
print("=" * 60)

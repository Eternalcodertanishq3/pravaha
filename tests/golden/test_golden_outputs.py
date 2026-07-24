"""Golden Dataset Benchmark Tests — Regression testing for agent logic and pipeline outputs."""

import pytest
from pravaha.swarm.orchestrator import SwarmOrchestrator
from pravaha.swarm.pipeline_dag import PipelineDAG, DAGNode


GOLDEN_TEST_CASES = [
    {
        "input": "Calculate 15 * 8",
        "expected_keywords": ["120"],
    },
    {
        "input": "Summarize python list comprehension syntax",
        "expected_keywords": ["list", "syntax"],
    },
]


def test_golden_dataset_structure():
    """Verify golden dataset case schema and structure."""
    assert len(GOLDEN_TEST_CASES) >= 2
    for case in GOLDEN_TEST_CASES:
        assert "input" in case
        assert "expected_keywords" in case
        assert isinstance(case["expected_keywords"], list)


def test_pipeline_dag_topological_sort():
    """Verify deterministic node execution ordering in DAG."""
    dag = PipelineDAG()
    dag.add_node("planner", "planner")
    dag.add_node("coder", "coder", dependencies=["planner"])

    ready = [n.name for n in dag.get_ready_nodes()]
    assert ready == ["planner"]

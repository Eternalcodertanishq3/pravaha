"""Tests for debug routes — replayer, step_debugger, trace_logger."""

from __future__ import annotations


class TestReplayer:
    def test_get_recording_empty(self) -> None:
        from pravaha.debug.replayer import Replayer
        replayer = Replayer()
        assert replayer.get_recording("nonexistent") is None

    def test_record_and_retrieve(self) -> None:
        from pravaha.debug.replayer import Replayer
        replayer = Replayer()
        replayer.start_recording()
        replayer.record("req-1", "test prompt", {"temp": 0.7})
        replayer.record_token("req-1", "hello")
        replayer.record_token("req-1", " world")
        recording = replayer.get_recording("req-1")
        assert recording is not None
        assert recording["prompt"] == "test prompt"
        assert recording["tokens"] == ["hello", " world"]


class TestStepDebugger:
    def test_get_step_info_empty(self) -> None:
        from pravaha.debug.step_debugger import StepDebugger
        debugger = StepDebugger()
        assert debugger.get_step_info("req-1", 0) is None

    def test_record_and_get(self) -> None:
        from pravaha.debug.step_debugger import StepDebugger, TokenStep
        debugger = StepDebugger()
        step = TokenStep(position=0, token_id=42, token_text="hello", logprob=-0.5)
        debugger.record_step("req-1", step)
        info = debugger.get_step_info("req-1", 0)
        assert info is not None
        assert info["token_text"] == "hello"
        assert info["logprob"] == -0.5


class TestTraceLogger:
    def test_get_trace_empty(self) -> None:
        from pravaha.debug.trace_logger import TraceLogger
        logger = TraceLogger()
        assert logger.get_trace("nonexistent") == []

    def test_log_and_get_traces(self) -> None:
        from pravaha.debug.trace_logger import TraceLogger
        logger = TraceLogger()
        logger.log("engine", "generate", {"tokens": 50})
        traces = logger.get_traces()
        assert len(traces) == 1
        assert traces[0]["component"] == "engine"

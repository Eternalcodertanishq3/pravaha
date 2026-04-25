"""Workers — All 20 worker agents."""

from pravaha.swarm.agents.workers.classifier_agent import ClassifierAgent
from pravaha.swarm.agents.workers.coder_agent import CoderAgent
from pravaha.swarm.agents.workers.critic_agent import CriticAgent
from pravaha.swarm.agents.workers.debugger_agent import DebuggerAgent
from pravaha.swarm.agents.workers.ensemble_agent import EnsembleAgent
from pravaha.swarm.agents.workers.expander_agent import ExpanderAgent
from pravaha.swarm.agents.workers.extractor_agent import ExtractorAgent
from pravaha.swarm.agents.workers.judge_agent import JudgeAgent
from pravaha.swarm.agents.workers.memory_agent import MemoryAgent
from pravaha.swarm.agents.workers.merger_agent import MergerAgent
from pravaha.swarm.agents.workers.narrator_agent import NarratorAgent
from pravaha.swarm.agents.workers.planner_agent import PlannerAgent
from pravaha.swarm.agents.workers.reasoning_agent import ReasoningAgent
from pravaha.swarm.agents.workers.refiner_agent import RefinerAgent
from pravaha.swarm.agents.workers.researcher_agent import ResearcherAgent
from pravaha.swarm.agents.workers.router_agent import RouterAgent
from pravaha.swarm.agents.workers.summarizer_agent import SummarizerAgent
from pravaha.swarm.agents.workers.tool_agent import ToolAgent
from pravaha.swarm.agents.workers.translator_agent import TranslatorAgent
from pravaha.swarm.agents.workers.validator_agent import ValidatorAgent

WORKER_AGENTS: dict[str, type] = {
    "planner": PlannerAgent,
    "researcher": ResearcherAgent,
    "coder": CoderAgent,
    "debugger": DebuggerAgent,
    "reasoning": ReasoningAgent,
    "critic": CriticAgent,
    "refiner": RefinerAgent,
    "summarizer": SummarizerAgent,
    "narrator": NarratorAgent,
    "expander": ExpanderAgent,
    "extractor": ExtractorAgent,
    "classifier": ClassifierAgent,
    "router": RouterAgent,
    "translator": TranslatorAgent,
    "ensemble": EnsembleAgent,
    "merger": MergerAgent,
    "judge": JudgeAgent,
    "memory": MemoryAgent,
    "tool": ToolAgent,
    "validator": ValidatorAgent,
}

__all__ = ["WORKER_AGENTS"]

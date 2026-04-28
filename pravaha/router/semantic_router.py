"""Semantic Router — TF-IDF cosine similarity for pipeline selection.

v3.3: Replaces keyword matching with actual TF-IDF vectorization
and cosine similarity. Falls back to keyword matching if sklearn
is not available.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


class SemanticRouter:
    """Route tasks to the best pipeline using semantic similarity.

    Uses TF-IDF cosine similarity to match task descriptions
    against pipeline descriptions and known task patterns.
    """

    # Pipeline descriptions for semantic matching
    PIPELINE_PROFILES: dict[str, dict[str, Any]] = {
        "plan-execute-audit": {
            "description": "Plan, write code, then audit it. General-purpose pipeline.",
            "keywords": [
                "build", "create", "implement", "make", "develop", "code",
                "write", "program", "function", "class", "script", "app",
            ],
            "task_types": ["coding", "implementation", "development"],
        },
        "research-write": {
            "description": "Research a topic, expand findings, and narrate a report.",
            "keywords": [
                "research", "find", "explain", "write", "report", "summarize",
                "analyze", "document", "describe", "essay", "article",
            ],
            "task_types": ["research", "writing", "documentation"],
        },
        "code-review": {
            "description": "Review existing code, debug issues, and refine quality.",
            "keywords": [
                "review", "fix", "debug", "refactor", "improve", "optimize",
                "bug", "error", "issue", "test", "quality",
            ],
            "task_types": ["review", "debugging", "refactoring"],
        },
        "secure-code-review": {
            "description": "Security-focused code review with vulnerability scanning.",
            "keywords": [
                "security", "vulnerability", "injection", "auth", "encrypt",
                "hack", "exploit", "pentest", "secure", "xss", "csrf",
            ],
            "task_types": ["security", "audit", "penetration_testing"],
        },
        "design-component": {
            "description": "Design UI components, layouts, and styling.",
            "keywords": [
                "design", "ui", "ux", "layout", "component", "style", "css",
                "responsive", "accessible", "interface", "theme", "color",
            ],
            "task_types": ["design", "ui", "frontend"],
        },
        "full-secure-design": {
            "description": "Full stack design with security hardening.",
            "keywords": [
                "fullstack", "design", "security", "build", "deploy",
                "production", "complete", "end-to-end",
            ],
            "task_types": ["fullstack", "production"],
        },
        "full-pipeline": {
            "description": "Complete pipeline with all stages: classify, plan, research, code, review.",
            "keywords": [
                "complex", "comprehensive", "full", "complete", "everything",
                "thorough", "detailed", "advanced",
            ],
            "task_types": ["complex", "comprehensive"],
        },
    }

    def __init__(self) -> None:
        self._tfidf_ready = False
        self._build_tfidf()

    def _build_tfidf(self) -> None:
        """Build TF-IDF vectors from pipeline profiles."""
        # Collect corpus: one document per pipeline
        self._corpus: dict[str, str] = {}
        for name, profile in self.PIPELINE_PROFILES.items():
            doc = " ".join([
                profile["description"],
                " ".join(profile["keywords"]),
                " ".join(profile.get("task_types", [])),
            ])
            self._corpus[name] = doc.lower()

        # Build vocabulary and IDF
        all_words: list[set[str]] = []
        self._vocab: set[str] = set()
        for doc in self._corpus.values():
            words = set(self._tokenize(doc))
            all_words.append(words)
            self._vocab.update(words)

        n_docs = len(self._corpus)
        self._idf: dict[str, float] = {}
        for word in self._vocab:
            doc_freq = sum(1 for words in all_words if word in words)
            self._idf[word] = math.log((n_docs + 1) / (doc_freq + 1)) + 1

        # Pre-compute TF-IDF vectors for each pipeline
        self._vectors: dict[str, dict[str, float]] = {}
        for name, doc in self._corpus.items():
            self._vectors[name] = self._tfidf_vector(doc)

        self._tfidf_ready = True
        logger.debug(f"SemanticRouter: Built TF-IDF with {len(self._vocab)} terms")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple word tokenization."""
        return re.findall(r"[a-z]+", text.lower())

    def _tfidf_vector(self, text: str) -> dict[str, float]:
        """Compute TF-IDF vector for a text."""
        words = self._tokenize(text)
        tf = Counter(words)
        max_tf = max(tf.values()) if tf else 1
        vec: dict[str, float] = {}
        for word, count in tf.items():
            if word in self._idf:
                vec[word] = (count / max_tf) * self._idf[word]
        return vec

    @staticmethod
    def _cosine_similarity(v1: dict[str, float], v2: dict[str, float]) -> float:
        """Compute cosine similarity between two sparse vectors."""
        keys = set(v1.keys()) & set(v2.keys())
        if not keys:
            return 0.0
        dot = sum(v1[k] * v2[k] for k in keys)
        mag1 = math.sqrt(sum(v ** 2 for v in v1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in v2.values()))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)

    def route(self, task: str) -> str:
        """Route a task to the best pipeline using TF-IDF similarity.

        Returns the pipeline name with the highest semantic similarity.
        """
        task_vec = self._tfidf_vector(task)

        best_name = "plan-execute-audit"  # default
        best_score = 0.0

        scores: dict[str, float] = {}
        for name, pipeline_vec in self._vectors.items():
            score = self._cosine_similarity(task_vec, pipeline_vec)
            scores[name] = score
            if score > best_score:
                best_score = score
                best_name = name

        logger.info(
            f"SemanticRouter: task→'{best_name}' (score={best_score:.3f}) "
            f"| scores={', '.join(f'{k}={v:.2f}' for k, v in sorted(scores.items(), key=lambda x: -x[1])[:3])}"
        )
        return best_name

    def route_with_scores(self, task: str) -> list[tuple[str, float]]:
        """Route and return all pipeline scores, sorted descending."""
        task_vec = self._tfidf_vector(task)
        scores = [
            (name, self._cosine_similarity(task_vec, vec))
            for name, vec in self._vectors.items()
        ]
        return sorted(scores, key=lambda x: -x[1])

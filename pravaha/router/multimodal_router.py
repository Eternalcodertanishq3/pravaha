"""Multimodal Router — Route text vs image vs video inputs."""

from __future__ import annotations
import logging
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class InputModality(Enum):
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    MIXED = auto()


class MultimodalRouter:
    """Detect input modality and route to the appropriate pipeline."""

    def detect_modality(self, content: Any) -> InputModality:
        if isinstance(content, str):
            return InputModality.TEXT
        if isinstance(content, list):
            has_image = any(isinstance(c, dict) and c.get("type") == "image_url" for c in content)
            has_text = any(isinstance(c, dict) and c.get("type") == "text" for c in content)
            if has_image and has_text:
                return InputModality.MIXED
            if has_image:
                return InputModality.IMAGE
        return InputModality.TEXT

    def route(self, content: Any) -> str:
        modality = self.detect_modality(content)
        if modality in (InputModality.IMAGE, InputModality.MIXED):
            return "vision_pipeline"
        return "text_pipeline"

"""Vision Detector — Detect images in multimodal inputs."""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ImageInput:
    url: str = ""
    base64_data: str = ""
    file_path: str = ""


class VisionDetector:
    """Detect and extract images from chat messages."""

    URL_PATTERN = re.compile(r'https?://\S+\.(png|jpg|jpeg|gif|webp)', re.IGNORECASE)

    def detect(self, messages: list[dict]) -> list[ImageInput]:
        images: list[ImageInput] = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:image"):
                            images.append(ImageInput(base64_data=url))
                        else:
                            images.append(ImageInput(url=url))
            elif isinstance(content, str):
                for match in self.URL_PATTERN.finditer(content):
                    images.append(ImageInput(url=match.group()))
        return images

    def has_images(self, messages: list[dict]) -> bool:
        return len(self.detect(messages)) > 0

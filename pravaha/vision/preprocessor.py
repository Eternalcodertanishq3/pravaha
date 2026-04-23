"""Vision Preprocessor — Resize and normalize images for model input."""

from __future__ import annotations
import base64
import io
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class VisionPreprocessor:
    """Preprocess images for vision-language models."""

    def __init__(self, max_size: int = 768) -> None:
        self.max_size = max_size

    def preprocess(self, image_data: str | bytes) -> Any:
        try:
            from PIL import Image
            if isinstance(image_data, str):
                if image_data.startswith("data:image"):
                    _, b64 = image_data.split(",", 1)
                    image_data = base64.b64decode(b64)
                else:
                    import httpx
                    resp = httpx.get(image_data)
                    image_data = resp.content
            img = Image.open(io.BytesIO(image_data))
            if max(img.size) > self.max_size:
                img.thumbnail((self.max_size, self.max_size))
            return img
        except ImportError:
            raise ImportError("Pillow required for vision. Install: pip install 'pravaha[vision]'")

    def to_tensor(self, image: Any) -> Any:
        try:
            from torchvision import transforms
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            return transform(image)
        except ImportError:
            raise ImportError("torchvision required. Install: pip install 'pravaha[vision]'")

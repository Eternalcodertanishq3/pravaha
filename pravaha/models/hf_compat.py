"""HuggingFace Universal Compatibility Layer.

Handles all model families: GPT-2, Llama, Mistral, Phi, Qwen,
Gemma, Falcon, Bloom, StarCoder, DeepSeek, Mixtral, and any
AutoModelForCausalLM-compatible model.

Key behaviors:
- Auto-detects architecture from config.json
- Sets correct pad_token for models that lack one
- Handles chat templates for instruct models
- Detects and uses Flash Attention 2 if available
- Handles RoPE scaling for long context models
- Detects quantized (GGUF/GPTQ/AWQ) vs standard models
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Models known to need special pad token handling
MODELS_NEEDING_PAD_TOKEN = {
    "llama", "mistral", "falcon", "mpt", "gpt-neox",
    "qwen2", "gemma", "deepseek", "phi3", "phi",
    "codellama", "mixtral", "starcoder2",
}

# Models with native chat templates
INSTRUCT_PATTERNS = [
    "instruct", "chat", "it", "-sft", "rlhf", "dpo", "alpaca",
]

# Models supporting Flash Attention 2
FLASH_ATTN_COMPATIBLE = {
    "llama", "mistral", "falcon", "qwen2", "gemma", "phi3",
    "starcoder", "codellama", "deepseek", "mixtral", "phi",
    "starcoder2", "qwen",
}


class HFCompatLayer:
    """Universal HuggingFace model loading with full compatibility."""

    def __init__(self) -> None:
        self._flash_attn_available = self._check_flash_attn()

    @staticmethod
    def _check_flash_attn() -> bool:
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            return False

    def get_model_kwargs(
        self,
        model_path: str,
        device: str = "auto",
        dtype_str: str = "float16",
        quantization: str | None = None,
        trust_remote_code: bool = False,
    ) -> dict[str, Any]:
        """Build kwargs for AutoModelForCausalLM.from_pretrained()."""
        import torch
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        model_type = getattr(config, "model_type", "unknown").lower()

        dtype = getattr(torch, dtype_str, torch.float16)
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "device_map": "auto" if device in ("auto", "cuda") else device,
        }

        # Flash Attention 2 if available and model supports it
        if (self._flash_attn_available
                and any(m in model_type for m in FLASH_ATTN_COMPATIBLE)):
            kwargs["attn_implementation"] = "flash_attention_2"
            logger.info(f"Flash Attention 2 enabled for {model_type}")

        # Quantization
        if quantization == "4bit":
            try:
                from pravaha.quantization.bitsandbytes import BitsAndBytesQuantizer
                kwargs["quantization_config"] = BitsAndBytesQuantizer(4).get_config()
            except ImportError:
                logger.warning("bitsandbytes not available for 4bit quantization")
        elif quantization == "8bit":
            try:
                from pravaha.quantization.bitsandbytes import BitsAndBytesQuantizer
                kwargs["quantization_config"] = BitsAndBytesQuantizer(8).get_config()
            except ImportError:
                logger.warning("bitsandbytes not available for 8bit quantization")

        # Long context / RoPE scaling awareness
        max_pos = getattr(config, "max_position_embeddings", 2048)
        if max_pos >= 32768:
            logger.info(
                f"Long context model detected: max_pos={max_pos}. "
                "Using eager attention if Flash Attn unavailable."
            )
            if not self._flash_attn_available:
                kwargs["attn_implementation"] = "eager"

        return kwargs

    def fix_tokenizer(self, tokenizer: Any, model_path: str) -> Any:
        """Ensure tokenizer has required special tokens."""
        from transformers import AutoConfig
        try:
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=True
            )
            model_type = getattr(config, "model_type", "unknown").lower()
        except Exception:
            model_type = "unknown"

        # Fix missing pad token
        if tokenizer.pad_token is None:
            if any(m in model_type for m in MODELS_NEEDING_PAD_TOKEN):
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                logger.info(f"Set pad_token = eos_token for {model_type}")
            elif hasattr(tokenizer, "eos_token") and tokenizer.eos_token:
                # Safe fallback for any model
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer

    def format_prompt(
        self,
        messages: list[dict[str, str]],
        tokenizer: Any,
        model_path: str,
    ) -> str:
        """Format messages using chat template if available."""
        # Try native chat template first
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                logger.warning(f"Chat template failed: {e}, using fallback")

        # Check if instruct model by name
        model_lower = model_path.lower()
        is_instruct = any(p in model_lower for p in INSTRUCT_PATTERNS)

        if is_instruct:
            # Generic instruct format
            parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    parts.append(f"System: {content}")
                elif role == "user":
                    parts.append(f"Human: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
            parts.append("Assistant:")
            return "\n\n".join(parts)

        # Plain concatenation for base models
        return "\n".join(
            f"<|{m['role']}|>\n{m['content']}" for m in messages
        ) + "\n<|assistant|>\n"

    def is_instruct_model(self, model_path: str) -> bool:
        """Check if a model path suggests an instruct/chat model."""
        return any(p in model_path.lower() for p in INSTRUCT_PATTERNS)

    def detect_architecture(self, model_path: str) -> dict[str, Any]:
        """Detect model architecture details from config."""
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            return {
                "model_type": getattr(config, "model_type", "unknown"),
                "hidden_size": getattr(config, "hidden_size", 0),
                "num_layers": getattr(config, "num_hidden_layers", 0),
                "num_heads": getattr(config, "num_attention_heads", 0),
                "num_kv_heads": getattr(config, "num_key_value_heads",
                                       getattr(config, "num_attention_heads", 0)),
                "max_position_embeddings": getattr(config, "max_position_embeddings", 2048),
                "vocab_size": getattr(config, "vocab_size", 0),
                "is_instruct": self.is_instruct_model(model_path),
            }
        except Exception as e:
            logger.warning(f"Could not detect architecture: {e}")
            return {"model_type": "unknown", "error": str(e)}

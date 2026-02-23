"""Tests for 8-bit and 4-bit quantization using bitsandbytes."""

import gc
import logging
import torch
import pytest

from pravaha.models.loader import ModelLoader

# Disable loader logging to keep test output clean
logging.getLogger("pravaha.models.loader").setLevel(logging.CRITICAL)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Quantization requires CUDA")
def test_quantization_memory_savings():
    """Verify that bitsandbytes reduces the GPU memory footprint of the loaded model."""
    loader = ModelLoader()
    model_name = "gpt2" # Using a small model to make the test run quickly

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping test.")
        return

    # 1. Base fp16 memory
    torch.cuda.empty_cache()
    model_fp16, _ = loader.load_model(model_name, dtype=torch.float16, device="cuda", quantization=None)
    mem_fp16 = torch.cuda.memory_allocated() / (1024**2) # MB
    
    # Cleanup memory
    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. 8-bit memory
    try:
        model_8bit, _ = loader.load_model(model_name, dtype=torch.float16, device="cuda", quantization="8bit")
        mem_8bit = torch.cuda.memory_allocated() / (1024**2) # MB
        
        del model_8bit
        gc.collect()
        torch.cuda.empty_cache()
    except ImportError:
        pytest.skip("bitsandbytes not installed")
        return

    # 3. 4-bit memory
    try:
        model_4bit, _ = loader.load_model(model_name, dtype=torch.float16, device="cuda", quantization="4bit")
        mem_4bit = torch.cuda.memory_allocated() / (1024**2) # MB
        
        del model_4bit
        gc.collect()
        torch.cuda.empty_cache()
    except ImportError:
        pytest.skip("bitsandbytes not installed")
        return

    print(f"\n--- Memory Footprint ({model_name}) ---")
    print(f"FP16 : {mem_fp16:.2f} MB")
    print(f"8-bit: {mem_8bit:.2f} MB")
    print(f"4-bit: {mem_4bit:.2f} MB")
    print("-" * 35)

    # 8-bit should be roughly half the size of FP16
    assert mem_8bit < mem_fp16, f"8-bit ({mem_8bit}MB) should use less memory than FP16 ({mem_fp16}MB)"
    
    # 4-bit should be roughly half the size of 8-bit
    assert mem_4bit < mem_8bit, f"4-bit ({mem_4bit}MB) should use less memory than 8-bit ({mem_8bit}MB)"
    
    print("✅ Quantization memory reductions verified successfully.")

if __name__ == "__main__":
    test_quantization_memory_savings()

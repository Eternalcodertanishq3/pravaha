import asyncio
import logging
import sys
import os

# Set PYTHONPATH
sys.path.append(os.getcwd())

from pravaha.engine_async import AsyncPravahaEngine
from pravaha.decoder.sampling import SamplingParams

async def test_prefix_sharing():
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    engine = AsyncPravahaEngine()
    
    prompt1 = "The quick brown fox jumps over the lazy dog. This is a very long sentence designed to fill multiple blocks in the KV-cache for testing purposes."
    prompt2 = "The quick brown fox jumps over the lazy dog. This is a very long sentence designed to fill multiple blocks in the KV-cache for testing purposes. Actually, it was a cat."
    
    print("\nStarting concurrent requests (prompt2 should share prefix with prompt1)...")
    
    # We create two concurrent generate tasks
    async def run_gen(idx, prompt):
        print(f"Request {idx} starting...")
        async for _ in engine.generate(prompt, SamplingParams(max_new_tokens=5)):
            pass
        print(f"Request {idx} finished.")

    await asyncio.gather(
        run_gen(1, prompt1),
        run_gen(2, prompt2)
    )

    engine.stop_background_loop()

if __name__ == "__main__":
    asyncio.run(test_prefix_sharing())

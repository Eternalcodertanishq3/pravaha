import asyncio
import logging
import sys
import os

# Set PYTHONPATH
sys.path.append(os.getcwd())

from pravaha.engine_async import AsyncPravahaEngine
from pravaha.decoder.sampling import SamplingParams

async def test_swapping():
    # Use higher log level to reduce console noise while keeping engine logs
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    logger = logging.getLogger("test_swapping")
    
    print("\n[TEST] Initializing Engine with 4 blocks (8 tokens each)...")
    engine = AsyncPravahaEngine()
    
    # Overriding to force extreme memory pressure
    engine.max_batch_size = 2
    engine.block_size = 8
    num_test_blocks = 4 # 32 tokens total
    
    from pravaha.kv_cache.paged_cache import PagedKVCache
    from pravaha.scheduler.scheduler import ContinuousScheduler
    
    engine.kv_cache = PagedKVCache.from_model_config(
        arch_config=engine.arch_config,
        num_blocks=num_test_blocks,
        block_size=engine.block_size,
        device=engine._device
    )
    engine.scheduler = ContinuousScheduler(
        num_blocks=num_test_blocks,
        block_size=engine.block_size,
        max_batch_size=engine.max_batch_size,
        max_seq_len=128
    )
    engine.decoder.kv_cache = engine.kv_cache

    # Each prompt is ~12-14 tokens (takes 2 blocks each)
    # Total blocks for 2 requests = 4. 0 left!
    prompt = "The quick brown fox jumps over the " 
    
    print("\n[TEST] Submitting REQ 1...")
    task1 = asyncio.create_task(consume_gen(engine, prompt, 1))
    
    await asyncio.sleep(0.5)
    
    print("\n[TEST] Submitting REQ 2... (Will fill all blocks)")
    task2 = asyncio.create_task(consume_gen(engine, prompt, 2))
    
    # Once both are running and they try to generate their 1st or 2nd token, 
    # they will hit the 16th token boundary (end of 2nd block) and trigger OOM.
    
    await asyncio.gather(task1, task2)
    engine.stop_background_loop()

async def consume_gen(engine, prompt, idx):
    print(f"[Req {idx}] started.")
    count = 0
    try:
        async for token in engine.generate(prompt, SamplingParams(max_new_tokens=10)):
            count += 1
            # print(f"[Req {idx}] token: {token}")
    except Exception as e:
        print(f"[Req {idx}] Error: {e}")
    print(f"[Req {idx}] finished after {count} tokens.")

if __name__ == "__main__":
    try:
        asyncio.run(test_swapping())
    except KeyboardInterrupt:
        pass

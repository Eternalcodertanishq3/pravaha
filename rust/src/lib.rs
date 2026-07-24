mod allocator;
mod prefix_trie;
mod stats;
mod token_bridge;
mod http_server;

use allocator::BlockAllocator;
use prefix_trie::PrefixTrie;
use stats::AllocatorStats;
use token_bridge::TokenBridge;
use pyo3::prelude::*;

#[pymodule]
fn pravaha_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BlockAllocator>()?;
    m.add_class::<PrefixTrie>()?;
    m.add_class::<AllocatorStats>()?;
    m.add_class::<TokenBridge>()?;
    Ok(())
}

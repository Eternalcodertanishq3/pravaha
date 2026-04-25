mod allocator;
mod prefix_trie;
mod stats;

use allocator::BlockAllocator;
use prefix_trie::PrefixTrie;
use stats::AllocatorStats;
use pyo3::prelude::*;

#[pymodule]
fn pravaha_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BlockAllocator>()?;
    m.add_class::<PrefixTrie>()?;
    m.add_class::<AllocatorStats>()?;
    Ok(())
}

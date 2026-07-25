#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_local_definitions)]

mod allocator;
mod prefix_trie;
mod stats;
mod token_bridge;
pub mod http_server;

use allocator::BlockAllocator;
use prefix_trie::PrefixTrie;
use stats::AllocatorStats;
use token_bridge::TokenBridge;
use http_server::start_server_bg;
use pyo3::prelude::*;

#[pymodule]
fn pravaha_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BlockAllocator>()?;
    m.add_class::<PrefixTrie>()?;
    m.add_class::<AllocatorStats>()?;
    m.add_class::<TokenBridge>()?;
    m.add_function(wrap_pyfunction!(start_server_bg, m)?)?;
    Ok(())
}

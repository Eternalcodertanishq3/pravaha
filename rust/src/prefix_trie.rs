use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// A single node in the prefix trie.
struct TrieNode {
    children: HashMap<u32, Arc<RwLock<TrieNode>>>,
    ref_count: usize,
    is_terminal: bool,
    block_id: Option<usize>,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            ref_count: 0,
            is_terminal: false,
            block_id: None,
        }
    }
}

/// Token-level prefix trie for KV cache sharing.
///
/// Uses Arc<RwLock<T>> for concurrent read access during continuous
/// batching. Multiple requests can read simultaneously to find
/// shared prefixes; only insertions require write locks.
#[pyclass]
pub struct PrefixTrie {
    root: Arc<RwLock<TrieNode>>,
    total_nodes: usize,
    total_hits: usize,
    total_misses: usize,
}

#[pymethods]
impl PrefixTrie {
    #[new]
    pub fn new() -> Self {
        Self {
            root: Arc::new(RwLock::new(TrieNode::new())),
            total_nodes: 1,
            total_hits: 0,
            total_misses: 0,
        }
    }

    /// Insert a token sequence and associate it with a block_id.
    pub fn insert(&mut self, tokens: Vec<u32>, block_id: usize) -> PyResult<()> {
        let mut current = Arc::clone(&self.root);

        for token in &tokens {
            let next = {
                let mut node = current.write().map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
                })?;

                if !node.children.contains_key(token) {
                    let new_node = Arc::new(RwLock::new(TrieNode::new()));
                    node.children.insert(*token, Arc::clone(&new_node));
                    self.total_nodes += 1;
                    new_node
                } else {
                    Arc::clone(&node.children[token])
                }
            };
            current = next;
        }

        // Mark terminal
        let mut terminal = current.write().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
        })?;
        terminal.is_terminal = true;
        terminal.ref_count += 1;
        terminal.block_id = Some(block_id);

        Ok(())
    }

    /// Find the longest matching prefix for a token sequence.
    /// Returns (matched_length, block_id) or (0, None) if no match.
    ///
    /// Uses read locks only — safe for concurrent access during
    /// continuous batching.
    pub fn longest_prefix_match(&mut self, tokens: Vec<u32>) -> PyResult<(usize, Option<usize>)> {
        let mut current = Arc::clone(&self.root);
        let mut best_length: usize = 0;
        let mut best_block_id: Option<usize> = None;

        for (i, token) in tokens.iter().enumerate() {
            let next = {
                let node = current.read().map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
                })?;

                if let Some(child) = node.children.get(token) {
                    Arc::clone(child)
                } else {
                    break;
                }
            };

            // Check if this is a terminal node
            {
                let node = next.read().map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
                })?;
                if node.is_terminal {
                    best_length = i + 1;
                    best_block_id = node.block_id;
                }
            }

            current = next;
        }

        if best_length > 0 {
            self.total_hits += 1;
        } else {
            self.total_misses += 1;
        }

        Ok((best_length, best_block_id))
    }

    /// Decrement ref count for a token sequence. If zero, the path
    /// can be garbage-collected in a future compaction pass.
    pub fn decrement_ref(&mut self, tokens: Vec<u32>) -> PyResult<bool> {
        let mut current = Arc::clone(&self.root);

        for token in &tokens {
            let next = {
                let node = current.read().map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
                })?;

                match node.children.get(token) {
                    Some(child) => Arc::clone(child),
                    None => return Ok(false),
                }
            };
            current = next;
        }

        let mut terminal = current.write().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
        })?;

        if terminal.ref_count > 0 {
            terminal.ref_count -= 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get trie statistics.
    pub fn stats(&self) -> PyResult<(usize, usize, usize)> {
        Ok((self.total_nodes, self.total_hits, self.total_misses))
    }

    /// Hit rate as a percentage.
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            return 0.0;
        }
        (self.total_hits as f64 / total as f64) * 100.0
    }

    /// Total nodes in the trie.
    pub fn node_count(&self) -> usize {
        self.total_nodes
    }
}

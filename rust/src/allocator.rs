use pyo3::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BlockState {
    Free,
    GPU,
    CPU,
}

impl IntoPy<PyObject> for BlockState {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            BlockState::Free => "free".into_py(py),
            BlockState::GPU => "gpu".into_py(py),
            BlockState::CPU => "cpu".into_py(py),
        }
    }
}

#[pyclass]
pub struct BlockAllocator {
    num_blocks: usize,
    free_blocks: Vec<usize>,
    ref_counts: Vec<usize>,
    states: Vec<BlockState>,
    last_touched: Vec<u64>,
}

#[pymethods]
impl BlockAllocator {
    #[new]
    pub fn new(num_blocks: usize) -> Self {
        let mut free_blocks = Vec::with_capacity(num_blocks);
        for i in (0..num_blocks).rev() {
            free_blocks.push(i);
        }

        Self {
            num_blocks,
            free_blocks,
            ref_counts: vec![0; num_blocks],
            states: vec![BlockState::Free; num_blocks],
            last_touched: vec![0; num_blocks],
        }
    }

    pub fn allocate(&mut self, num_required: usize) -> PyResult<Vec<usize>> {
        if self.free_blocks.len() < num_required {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Out of memory blocks"));
        }

        let mut allocated = Vec::with_capacity(num_required);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        for _ in 0..num_required {
            if let Some(block_id) = self.free_blocks.pop() {
                self.ref_counts[block_id] = 1;
                self.states[block_id] = BlockState::GPU;
                self.last_touched[block_id] = now;
                allocated.push(block_id);
            }
        }

        Ok(allocated)
    }

    pub fn free(&mut self, block_id: usize) -> PyResult<()> {
        if block_id >= self.num_blocks {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid block_id"));
        }

        if self.ref_counts[block_id] > 0 {
            self.ref_counts[block_id] -= 1;
            if self.ref_counts[block_id] == 0 {
                self.states[block_id] = BlockState::Free;
                self.free_blocks.push(block_id);
            }
        }

        Ok(())
    }

    pub fn touch(&mut self, block_id: usize) -> PyResult<()> {
        if block_id >= self.num_blocks {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid block_id"));
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.last_touched[block_id] = now;
        Ok(())
    }

    pub fn swap_out(&mut self, block_id: usize) -> PyResult<()> {
        if block_id >= self.num_blocks {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid block_id"));
        }
        if self.states[block_id] == BlockState::GPU {
            self.states[block_id] = BlockState::CPU;
        }
        Ok(())
    }

    pub fn swap_in(&mut self, block_id: usize) -> PyResult<()> {
        if block_id >= self.num_blocks {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid block_id"));
        }
        if self.states[block_id] == BlockState::CPU {
            self.states[block_id] = BlockState::GPU;
        }
        Ok(())
    }

    pub fn get_block_state(&self, block_id: usize) -> PyResult<BlockState> {
        if block_id >= self.num_blocks {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid block_id"));
        }
        Ok(self.states[block_id])
    }

    pub fn get_ref_count(&self, block_id: usize) -> PyResult<usize> {
        if block_id >= self.num_blocks {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid block_id"));
        }
        Ok(self.ref_counts[block_id])
    }

    pub fn increment_ref(&mut self, block_id: usize) -> PyResult<()> {
        if block_id >= self.num_blocks {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid block_id"));
        }
        if self.states[block_id] == BlockState::Free {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Cannot increment ref on free block"));
        }
        self.ref_counts[block_id] += 1;
        Ok(())
    }

    pub fn compact(&mut self) -> PyResult<()> {
        // Simple compaction: just ensure free_blocks list is somewhat sorted or consistent
        // In a fixed-page allocator, fragmentation isn't usually an issue unless we need
        // contiguous blocks for some reason (which we don't for PagedAttention).
        // For now, we'll just sort the free list to keep it predictable.
        self.free_blocks.sort_by(|a, b| b.cmp(a)); // Store in reverse so poppers get low IDs
        Ok(())
    }

    pub fn get_lru_block(&self) -> PyResult<Option<usize>> {
        let mut oldest_time = u64::MAX;
        let mut lru_block = None;

        for i in 0..self.num_blocks {
            if self.states[i] == BlockState::GPU && self.ref_counts[i] > 0 {
                if self.last_touched[i] < oldest_time {
                    oldest_time = self.last_touched[i];
                    lru_block = Some(i);
                }
            }
        }

        Ok(lru_block)
    }

    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }
}
